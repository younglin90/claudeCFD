# An acoustic-convective splitting-based approach for the Kapila two-phase flow model

M.F.P. ten Eikeldera,b,<sup>∗</sup> , F. Daudea,c, B. Koren<sup>b</sup> , A.S. Tijsseling<sup>b</sup>

<sup>a</sup>EDF R&D, AMA, 7 boulevard Gaspard Monge 91120, Palaiseau, France <sup>b</sup>Eindhoven University of Technology, Department of Mathematics and Computer Science, P.O. Box 513, 5600 MB Eindhoven, The Netherlands

c IMSIA, UMR EDF-CNRS-CEA-ENSTA 9219, Universit´e Paris Saclay, 828 Boulevard des Mar´echaux 91762 Palaiseau, France

# Abstract

In this paper we propose a new acoustic-convective splitting-based numerical scheme for the Kapila fiveequation two-phase flow model. The splitting operator decouples the acoustic waves and convective waves. The resulting two submodels are alternately numerically solved to approximate the solution of the entire model. The Lagrangian form of the acoustic submodel is numerically solved using an HLLC-type Riemann solver whereas the convective part is approximated with an upwind scheme. The result is a simple method which allows for a general equation of state. Numerical computations are performed for standard twophase shock tube problems. A comparison is made with a non-splitting approach. The results are in good agreement with reference results and exact solutions.

Keywords: Two-phase compressible flows, Splitting-based method, Finite-volume method, HLLC scheme, Shock tube

## 1. Introduction

Compressible two-phase and two-fluid flow phenomena arise in many natural features and industrial applications. Examples are groundwater flow, surface wave impacts, oil slicks, water-air flows, shock-bubble interaction and (condensation induced) water hammer phenomena. The study of two-phase flow is a challenging research area which is of interest to both engineers and scientists.

Various models can be used to describe two-phase flows. Many of these models can be classified as two-fluid models, or homogeneous models. Among the two-fluid flow models, which are generally considered as the most complete, the model of Baer and Nunziato [\[1\]](#page-27-0) is one of the best known. This model consists of equations for each of the two fluids' mass, momentum, energy, and of an equation describing the topology of the two-fluid interface. Romenski et al. [\[2\]](#page-28-0) proposed a seven-equation model for two-phase compressible flow which can be written in Baer-Nunziato form in the heat flux relaxation limit. Due to the complexity of the seven-equation models, linked to their large number of different waves [\[3](#page-28-1)[–14\]](#page-28-2), reduced models with less equations have been proposed.

The five-equation models form an important class of reduced models. The original five-equation twophase flow model of Kapila et al. [\[15\]](#page-28-3) has been derived from the two-fluid flow model of Baer and Nunziato. To study pure interface problems the model of Allaire et al. [\[16–](#page-28-4)[18\]](#page-28-5) can be used. The model of Kapila et al., describing inviscid, non-heat-conducting, compressible two-fluid flow, allows for mixtures. To model phase

Present address: Delft University of Technology, Department of Mechanical, Maritime and Materials Engineering, P.O. Box 5, 2600 AA Delft, The Netherlands

Email address: m.f.p.teneikelder@tudelft.nl (M.F.P. ten Eikelder)

<sup>∗</sup>Corresponding author.

transitions, the five-equation model has been extended by taking temperature and chemical potential relaxation effects into account [\[19\]](#page-28-6). Murrone and Guillard [\[20\]](#page-28-7) give an analysis of the five-equation model and indicate that the five-equation model is a good approximation of the seven-equation two-fluid model. Kreeft and Koren [\[21\]](#page-28-8) propose a new formulation of the five-equation model, in which the topological equation is replaced by an energy equation. An Osher-type approximation is used for the evaluation of the fluxes and the energy-exchange term in the discretized system. Ahmed et al. [\[22\]](#page-28-9) use a central upwind scheme for the new formulation to study shock-bubble interaction problems. Daude et al. [\[17\]](#page-28-10) present computations with the original five-equation model of Kapila et al. using an HLLC-type scheme in the context of an Arbitrary Lagrangian-Eulerian formulation.

Serious difficulties are posed by the non-conservative terms in the topology equation of the five-equation model. In particular, (i) approximating the term containing velocity divergence, (ii) performing shock computations with a non-conservative model and (iii) ensuring volume fraction positivity [\[23–](#page-28-11)[25\]](#page-28-12) is difficult. Several approaches have been suggested to circumvent these issues. Abgrall and Perrier [\[26\]](#page-28-13) present, using probabilistic multiscale interpretation of multiphase flows, a locally conservative scheme to tackle the issues. Saurel-Petitpas-Berry [\[23,](#page-28-11) [27\]](#page-28-14) propose to relax the pressure equilibrium assumption and obtain a non-conservative hyperbolic six-equation model which simplifies numerical resolution. Jiang et al. [\[28\]](#page-28-15) use this six-equation approach with a novel mass transfer between liquid and vapor.

The aim of the present paper is to propose an acoustic-convective splitting-based numerical method for the five-equation two-phase flow model. Due to its simplicity, the original five-equation model of Kapila et al., without any relaxation or modification, is considered. Furthermore, the speed of sound of this model corresponds to the Wood speed of sound which is known to be in good agreement with the experimental data obtained at moderate frequencies of sound (pressure disturbance) in air-water mixtures. The present approach is inspired by the Lagrange-Projection-like scheme originally proposed for the Euler equations of gas dynamics, by Chalons et al. [\[29\]](#page-28-16). In this paper a method similar to that from [\[29\]](#page-28-16) is extended to the full two-phase five-equation model. Related work of the authors about the splitting approach has been presented in [\[30\]](#page-28-17). Our scheme uses an HLLC-type scheme for the acoustic model and a classical upwind scheme for the convective model. Conservation of mass, momentum, energy and partial mass, as well as the positivity of the volume fraction and the mass fraction are ensured. The advantages of the proposed approach are (i) its simplicity and (ii) its accurate capturing at shock waves and (iii) the potential to deal with low-Mach number flows. Approximate Godunov approaches and direct approaches may lead to inaccuracies at highly subsonic flows. By using a splitting operator these inaccuracies can be prevented [\[29\]](#page-28-16). Furthermore, unlike Osher-type schemes [\[21\]](#page-28-8), the current approach can deal with a general equation of state (just like the direct approach from [\[17\]](#page-28-10)). A similar idea has been proposed by Huber et al. [\[31\]](#page-28-18). They use a compressible projection method with a level-set method describing the interface motion to study the interaction of an ultrasound wave with a bubble.

The paper is organized as follows. In Section [2](#page-1-0) the five-equation two-phase Kapila et al. flow model is shortly rehearsed. The novel acoustic-convective splitting scheme is presented in Section [3.](#page-3-0) The numerical scheme is assessed for shock-tube problems in Section [4,](#page-9-0) and a comparison with the direct approach is made in terms of accuracy, efficiency and robustness. Conclusions are drawn in Section [5.](#page-27-1)

#### 2. Two-phase flow model

The five-equation model of Kapila et al. [\[15\]](#page-28-3) describes the dynamics of inviscid two-phase flows evolving in mechanical equilibrium (i.e. equilibrium of velocity and pressure is assumed across the fluid interface). The model consists of four balance equations for conservative quantities: two for mass (bulk mass and mass of one of the two phases), one for the bulk momentum and one for the bulk total energy. The fifth equation is a topological equation, of non-conservative type, which describes the evolution of the volume fraction. In one dimension, the governing equations read:

$$\partial_t \rho + \partial_x \left( \rho u \right) = 0,$$
 (1a)

$$\partial_t(\rho u) + \partial_x(\rho u^2 + p) = 0,$$
 (1b)

$$\partial_t(\rho E) + \partial_x(\rho E u + p u) = 0,$$
 (1c)

$$\partial_t(\alpha_1 \rho_1) + \partial_x(\alpha_1 \rho_1 u) = 0, \tag{1d}$$

$$\partial_t \alpha_1 + u \partial_x \alpha_1 + K \partial_x u = 0,$$
 (1e)

where t is the time, x the spatial coordinate, ρ the mixture density, u the bulk velocity, p the pressure and E the mixture total specific energy. The interfacial variable K is specified later. The variable αk, k = 1, 2, represents the volume fraction of phase k, with the saturation constraint α<sup>1</sup> + α<sup>2</sup> = 1, and ρ<sup>k</sup> denotes the density of phase k. In terms of separated fluid variables, the bulk density is given by

$$\rho = \alpha_1 \rho_1 + \alpha_2 \rho_2. \tag{2}$$

We define the mass fraction Y<sup>k</sup> of phase k as ρY<sup>k</sup> = αkρk. The entropy equations, i.e.:

$$\partial_t(\alpha_k \rho_k s_k) + \partial_x(\alpha_k \rho_k s_k u) = 0, \tag{3}$$

with s<sup>k</sup> the specific entropy of phase k, complement the model in absence of shocks [\[20\]](#page-28-7). All the dissipative effects are neglected (inviscid, non-heat conducting flow is considered) and thus it can be written as

$$\frac{\mathrm{D}s_k}{\mathrm{D}t} = 0,\tag{4}$$

with the Lagrangian derivative D/Dt := ∂<sup>t</sup> + u∂x. The total specific energy of the mixture is given by:

$$\rho E = \alpha_1 \rho_1 E_1 + \alpha_2 \rho_2 E_2,\tag{5}$$

where the total specific energy of each of the two phases is

$$E_k = e_k + \frac{1}{2}u^2, (6)$$

with e<sup>k</sup> the internal specific energy of phase k. The bulk internal specific energy is given by

$$\rho e = \alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2,\tag{7}$$

and hence,

$$E = e + \frac{1}{2}u^2. \tag{8}$$

In the present paper, the model is completed with the stiffened gas (SG) equation of state (EOS) for each phase:

$$p = \rho_k (e_k - \eta_k)(\gamma_k - 1) - \gamma_k \pi_k, \tag{9}$$

where the pressure equilibrium across the interface is used. The ratio of specific heats γk, stiffness π<sup>k</sup> and energies at a reference state η<sup>k</sup> are characteristic constants of the thermodynamic behavior of fluid k. Expression [\(9\)](#page-2-0) reduces to the perfect gas (PG) EOS when π<sup>k</sup> and η<sup>k</sup> is equal to zero whereas a large value of π<sup>k</sup> implies a near-incompressible behavior [\[32\]](#page-28-19). The SG EOS parameters are determined by shock wave Hugoniot curves [\[33](#page-28-20)[–35\]](#page-29-0). This EOS is often used as a reasonable approximation for both liquids and gases under high pressure conditions [\[12,](#page-28-21) [13,](#page-28-22) [17,](#page-28-10) [21,](#page-28-8) [36,](#page-29-1) [37\]](#page-29-2). The EOS allows the determination of the speed of sound of each single phase

$$c_k^2 \equiv \frac{p - \rho_k^2 \partial_{\rho_k} e_k}{\rho_k^2 \partial_{\rho} e_k} = \gamma_k \frac{p + \pi_k}{\rho_k}.$$
 (10)

The interfacial variable in the topology equation [\(1e\)](#page-2-1) is given by

$$K = \left(\rho_1 c_1^2 - \rho_2 c_2^2\right) / \left(\frac{\rho_1 c_1^2}{\alpha_1} + \frac{\rho_2 c_2^2}{\alpha_2}\right). \tag{11}$$

The internal specific energy of the mixture satisfies

$$\rho e = p \left( \frac{\alpha_1}{\gamma_1 - 1} + \frac{\alpha_2}{\gamma_2 - 1} \right) + \alpha_1 \left( \frac{\gamma_1}{\gamma_1 - 1} \pi_1 + \rho_1 \eta_1 \right) + \alpha_2 \left( \frac{\gamma_2}{\gamma_2 - 1} \pi_2 + \rho_2 \eta_2 \right). \tag{12}$$

The five-equation model [\(1\)](#page-2-2) is hyperbolic and admits the wave speeds [\[20\]](#page-28-7)

$$\lambda_1 = u - c, \quad \lambda_{2,3,4} = u, \quad \lambda_5 = u + c,$$
 (13)

with c the mixture speed of sound which obeys the Wood formula [\[38\]](#page-29-3):

$$\frac{1}{\rho c^2} = \frac{\alpha_1}{\rho_1 c_1^2} + \frac{\alpha_2}{\rho_2 c_2^2}. (14)$$

The characteristic fields associated with the eigenvalues λ2,3,<sup>4</sup> are linearly degenerate (LD) and the other two fields are genuinely nonlinear (GNL) [\[20\]](#page-28-7).

#### 3. Numerical scheme

A novel splitting-based numerical scheme is presented, leading to two operators: one associated with the pressure and the other with the advection. The two submodels are referred to as acoustic and convective, respectively, in the sequel. First, the treatment of the acoustic submodel is discussed for which a simple and robust HLLC-type Riemann solver is used. Next, the upwind scheme for the convective submodel is given.

## 3.1. The splitting approach

The five-equation model deals with two kinds of wave speeds associated with its eigenvalues, i.e. the GNL waves are linked to acoustic pressure waves whereas the LD wave is connected to the convective velocity. In certain situations such as subsonic flows, the ratio between these two speeds can be large, leading to inaccuracies when using approximate Godunov approaches. In order to decouple acoustic and convective phenomena, a splitting operator is proposed. This splitting is inspired by the one proposed by Chalons et al. [\[29\]](#page-28-16) for the Euler equations of gas dynamics.

By using product-rule arguments the Kapila five-equation model [\(1\)](#page-2-2) is split into (i) the acoustic system:

$$\partial_t \rho + \rho \partial_x u = 0, \tag{15a}$$

$$\partial_t(\rho u) + \rho u \partial_x u + \partial_x p = 0, \tag{15b}$$

$$\partial_t(\rho E) + \rho E \partial_x u + \partial_x(pu) = 0, \tag{15c}$$

$$\partial_t Y_1 = 0, (15d)$$

$$\partial_t \alpha_1 + K \partial_x u = 0, \tag{15e}$$

and (ii) the convective system:

$$\partial_t \rho + u \partial_x \rho = 0, \tag{16a}$$

$$\partial_t(\rho u) + u\partial_x(\rho u) = 0,$$
 (16b)

$$\partial_t(\rho E) + u\partial_x(\rho E) = 0, \tag{16c}$$

$$\partial_t Y_1 + u \partial_x Y_1 = 0, \tag{16d}$$

$$\partial_t \alpha_1 + u \partial_x \alpha_1 = 0, \tag{16e}$$

where the evolution of the mass fraction, Eqs. (15d) and (16d), follows from Eqs. (1a) and (1d). The corresponding entropy equations of the acoustic and convective systems are respectively:

$$\partial_t s_k = 0,$$
 (17a)  
 $\partial_t s_k + u \partial_x s_k = 0.$  (17b)

$$\partial_t s_k + u \partial_x s_k = 0. \tag{17b}$$

Basically, the splitting decouples the Lagrangian derivative terms from the remaining terms. Therefore, the convective system can be written as DQ/Dt = 0 for  $Q \in \{\rho, \rho u, \rho E, \rho Y_1, \alpha_1\}$ . Now, the acoustic system contains all the pressure terms and the interfacial term of the topological equation (1e). Note that this interfacial term includes the spatial derivative of velocity and is therefore included in the acoustic system. The splitting step is first-order accurate in time. A higher-order temporal accuracy can be obtained, e.g. for second-order accuracy by employing Strang splitting [39].

The numerical solution of (1) consists of successively approximating the solution of the acoustic system and the convective system. By denoting the temporal step size with  $\Delta t$ , the mesh width with  $\Delta x$ , the fluid state at time  $n\Delta t$  and position  $j\Delta x$  with  $\mathbf{Q}_{i}^{n} \equiv (\rho, \rho u, \rho E, \rho Y_{1}, \alpha_{1})_{i}^{n}$ , and an intermediate time level with n+1-, the approximation within one time step reads:

- Update Q<sup>n</sup><sub>j</sub> to Q<sup>n+1-</sup><sub>j</sub> by time marching the acoustic system (15) with step size Δt;
   Update Q<sup>n+1-</sup><sub>j</sub> to Q<sup>n+1</sup><sub>j</sub> by time marching the convective system (16) with step size Δt.

The choice of numerically solving the submodels in this order is linked to the velocity approximation: the velocity of the acoustic system is used for the determination of the convective velocity in order to ensure the conservation of mass, momentum, energy and partial masses as it is detailed in Section 3.6. The details of each step are given in Sections 3.3 and 3.4.

#### 3.2. Mathematical analysis of the two submodels

The five-equation model (1) can be cast into the primitive form

$$\partial_t \mathbf{W} + \mathbf{B}(\mathbf{W}) \partial_x \mathbf{W} = \mathbf{0},\tag{18}$$

and the primitive form of the subsystems (15)-(16) reads: (i) for the acoustic system:

$$\partial_t \mathbf{W} + \mathbf{A}(\mathbf{W}) \partial_x \mathbf{W} = \mathbf{0},\tag{19}$$

and (ii) for the convective system:

$$\partial_t \mathbf{W} + \mathbf{C}(\mathbf{W}) \partial_x \mathbf{W} = \mathbf{0}, \tag{20}$$

where

$$\mathbf{B}(\mathbf{W}) = \mathbf{A}(\mathbf{W}) + \mathbf{C}(\mathbf{W}),\tag{21}$$

with

$$\mathbf{W} = \begin{pmatrix} \rho \\ u \\ p \\ Y_1 \\ \alpha_1 \end{pmatrix}, \quad \mathbf{A}(\mathbf{W}) = \begin{pmatrix} 0 & \rho & 0 & 0 & 0 \\ 0 & 0 & 1/\rho & 0 & 0 \\ 0 & \rho c^2 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & K & 0 & 0 & 0 \end{pmatrix}, \quad \mathbf{C}(\mathbf{W}) = u \, \mathbf{I}_5, \tag{22}$$

where  $\mathbf{I}_d$  is the identity matrix in  $\mathbb{R}^{d\times d}$ . The derivation of the pressure equation is straightforward and can be found in [20, 21, 40]. This casting reveals that the matrix **B** splits into an acoustic part **A** and a convective part C. The eigenvalues of the full system  $(\lambda_k)$  split also into an acoustic part  $(\lambda_k^a)$  and a convective part  $(\lambda_k^c)$  as  $\lambda_k = \lambda_k^a + \lambda_k^c$  with

$$\lambda_1^a = -c, \quad \lambda_{2,3,4}^a = 0, \quad \lambda_5^a = c,$$

$$\lambda_1^c = u, \quad \lambda_{2,3,4}^c = u, \quad \lambda_5^c = u.$$
(23)

The characteristic fields associated with the convective submodel are obviously LD. Concerning the acoustic submodel, the fields associated with the middle wave  $\lambda_{2,3,4}^a = 0$  are LD. The other two waves, associated with  $\lambda_1^a = -c, \lambda_5^a = c$ , can be shown, by using a similar argument as Murrone et al. [20], to be GNL in the non-isobaric case and LD in the isobaric case.

# 3.3. Numerical solution of the acoustic submodel

## 3.3.1. Lagrangian formulation

Introducing the specific volume  $\tau = 1/\rho$  and taking  $\{\tau, u, E, Y_1, \alpha_1\}$  as the set of variables, the acoustic system can be cast into the form

$$\partial_t \tau - \tau \partial_x u = 0, \tag{24a}$$

$$\partial_t u + \tau \partial_x p = 0, (24b)$$

$$\partial_t E + \tau \partial_x (pu) = 0, \tag{24c}$$

$$\partial_t Y_1 = 0, \tag{24d}$$

$$\partial_t \alpha_1 + \rho K \tau \partial_x u = 0. \tag{24e}$$

The Eqs. (24a)-(24c) describe the bulk fluid, and the Eqs. (24d)-(24e) describe the evolution of the fraction variables, which are specific for the five-equation two-phase flow model. The second term of each equation (except the fourth) contains the operator  $\tau \partial_x$ . As in [29], for  $t \in [t^n, t^n + \Delta t)$  we approximate  $\tau(x, t)\partial_x$  by  $\tau(x, t^n)\partial_x$ , where the time level is  $t^n = n\Delta t$  with time step  $\Delta t$ . We then introduce the mass variable m by  $dm = \rho(x, t^n)dx$ . The Lagrangian system

$$\partial_t \tau - \partial_m u = 0, (25a)$$

$$\partial_t u + \partial_m p = 0, (25b)$$

$$\partial_t E + \partial_m(pu) = 0, (25c)$$

$$\partial_t Y_1 = 0, \tag{25d}$$

$$\partial_t \alpha_1 + \rho K \partial_m u = 0, \tag{25e}$$

is a first-order in time approximation of (24). This system has the eigenvalues

$$(\lambda_1^a)^{\mathcal{L}ag} = -\rho c, \quad (\lambda_{2,3,4}^a)^{\mathcal{L}ag} = 0, \quad (\lambda_5^a)^{\mathcal{L}ag} = \rho c \tag{26}$$

and associated eigenvectors

$$(\mathbf{v}_{1}^{a})^{\mathcal{L}ag} = \begin{pmatrix} -1 \\ -\rho c \\ (\rho c)^{2} \\ 0 \\ \rho K \end{pmatrix}, \quad (\mathbf{v}_{2}^{a})^{\mathcal{L}ag} = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad (\mathbf{v}_{3}^{a})^{\mathcal{L}ag} = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix},$$

$$(\mathbf{v}_{4}^{a})^{\mathcal{L}ag} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}, \quad (\mathbf{v}_{5}^{a})^{\mathcal{L}ag} = \begin{pmatrix} -1 \\ \rho c \\ (\rho c)^{2} \\ 0 \\ \rho K \end{pmatrix}.$$

$$(27)$$

It can be written in the following vectorial form:

$$\partial_t \mathbf{Q}^{\mathcal{L}ag} + \partial_m \mathcal{F}^{\mathcal{L}ag}(\mathbf{Q}^{\mathcal{L}ag}) + \mathcal{B}^{\mathcal{L}ag}(\mathbf{Q}^{\mathcal{L}ag}) \, \partial_m u = \mathbf{0}, \tag{28}$$

where

$$\mathbf{Q}^{\mathcal{L}ag} = (\tau, u, E, Y_1, \alpha_1)^T, \tag{29a}$$

$$\mathcal{F}^{\mathcal{L}ag}(\mathbf{Q}^{\mathcal{L}ag}) = (-u, p, pu, 0, 0)^T, \tag{29b}$$

$$\mathcal{B}^{\mathcal{L}ag}(\mathbf{Q}^{\mathcal{L}ag}) = (0, 0, 0, 0, \rho K)^{T}.$$
(29c)

The superscript Lag is used for the variables in the Lagrangian system. The term F <sup>L</sup>ag is a conservative flux and the latter is the non-conservative term. System [\(28\)](#page-5-6)-[\(29\)](#page-6-0) is numerically approximated in the following.

#### 3.3.2. HLLC-type solver for the acoustic submodel in Lagrangian coordinates

An HLLC-type Riemann solver [\[41\]](#page-29-6) is used to solve the acoustic system [\(28\)](#page-5-6)-[\(29\)](#page-6-0). The finite-volume approximation of the Eqs. [\(28\)](#page-5-6)-[\(29\)](#page-6-0) on each mesh element [xj−1/2, xj+1/2] follows from integration over the mesh element and assuming a constant density in the m variable and constant interfacial term in each element, and reads

$$\partial_{t}(\left(\mathbf{Q}^{\mathcal{L}ag}\right)_{j}) + \frac{1}{\Delta m_{j}} \left(\left(\mathbf{F}^{\mathcal{L}ag}\right)_{j+1/2}^{\mathrm{HLLC}} - \left(\mathbf{F}^{\mathcal{L}ag}\right)_{j-1/2}^{\mathrm{HLLC}}\right) + \frac{1}{\Delta m_{j}} \mathcal{B}^{\mathcal{L}ag} \left(\left(\mathbf{Q}^{\mathcal{L}ag}\right)_{j}\right) \left(u_{j+1/2}^{*} - u_{j+1/2}^{*}\right) = 0,$$
(30)

with ∆m<sup>j</sup> = ρ n <sup>j</sup> ∆x. In this paper we employ the classical finite-volume notation in which subscript j refers to a cell average and j + 1/2 to a cell boundary. The HLLC-type numerical flux vector F <sup>L</sup>agHLLC , which approximates F <sup>L</sup>ag Q<sup>L</sup>ag , is obtained by applying the HLLC-type relations across the three different waves with eigenvalues [\(26\)](#page-5-7), see Figure [1.](#page-6-1) Using [\(27\)](#page-5-8) we see that the velocity and pressure are the Riemann

![](_page_6_Figure_9.jpeg)

Figure 1: The different states Q Lag L , Q Lag,∗ L , Q Lag,∗ R , Q Lag R and wave speeds −aj+1/2, 0, aj+1/<sup>2</sup> in the Riemann problem.

invariants of the LD middle wave. The HLLC-type relations across the left and right waves for the momentum equation are given by

$$p_{j+1/2}^* = p_j - a_{j+1/2} \left( u_{j+1/2}^* - u_j \right), \tag{31a}$$

$$p_{j+1/2}^* = p_{j+1} + a_{j+1/2} \left( u_{j+1/2}^* - u_{j+1} \right). \tag{31b}$$

where the acoustic impedance aj+1/<sup>2</sup> at the interface is estimated using the direct computation of the eigenvalues of the acoustic submodel:

$$a_{j+1/2} = \max(\rho_j c_j, \rho_{j+1} c_{j+1}). \tag{32}$$

This leads to a single-state HLLC numerical flux-vector:

$$\left(\mathbf{F}^{\mathcal{L}ag}\right)_{j+1/2}^{\text{HLLC}} = \left(-u^*, p^*, p^*u^*, 0, 0\right)_{j+1/2},$$
 (33)

where

$$u_{j+1/2}^* = \frac{u_j + u_{j+1}}{2} + \frac{p_j - p_{j+1}}{2a_{j+1/2}},$$
(34a)

$$p_{j+1/2}^* = \frac{p_j + p_{j+1}}{2} + \frac{a_{j+1/2}}{2} (u_j - u_{j+1}).$$
(34b)

The interfacial term of the topology equation is approximated at first-order by

$$K_j^n \left( u_{j+1/2}^* - u_{j-1/2}^* \right).$$
 (35)

Summarizing and using an explicit forward Euler time step, the update formula for the discretized acoustic system reads:

$$(\mathbf{Q}^{\mathcal{L}ag})_{j}^{n+1-} = (\mathbf{Q}^{\mathcal{L}ag})_{j}^{n} - \frac{\Delta t}{\rho_{j}^{n} \Delta x} \left( \left( \mathbf{F}^{\mathcal{L}ag} \right)_{j+1/2}^{\text{HLLC},n} - \left( \mathbf{F}^{\mathcal{L}ag} \right)_{j-1/2}^{\text{HLLC},n} \right) - K_{j}^{n} \frac{\Delta t}{\Delta x} \left( \left( \mathcal{H}^{\mathcal{L}ag} \right)_{j+1/2}^{n} - \left( \mathcal{H}^{\mathcal{L}ag} \right)_{j-1/2}^{n} \right),$$
(36)

where

$$\left(\mathcal{H}^{\mathcal{L}ag}\right)^{T} = (0, 0, 0, 0, u^{*}). \tag{37}$$

The numerical experiments in section 4 employ this update formula.

#### 3.3.3. Update of the acoustic submodel in Eulerian variables

The update formulae for the discretized acoustic system in terms of the Eulerian variables from (1) are a reformulation of those in (36)-(37) and read:

$$R_j \rho_j^{n+1-} = \rho_j^n, \tag{38a}$$

$$R_{j} \left(\rho u\right)_{j}^{n+1-} = \left(\rho u\right)_{j}^{n} - \frac{\Delta t}{\Delta x} \left(p_{j+1/2}^{*} - p_{j-1/2}^{*}\right), \tag{38b}$$

$$R_{j} (\rho E)_{j}^{n+1-} = (\rho E)_{j}^{n} - \frac{\Delta t}{\Delta x} \left( p_{j+1/2}^{*} u_{j+1/2}^{*} - p_{j-1/2}^{*} u_{j-1/2}^{*} \right), \tag{38c}$$

$$R_j (\rho Y_1)_j^{n+1-} = (\rho Y_1)_j^n,$$
 (38d)

$$(\alpha_1)_j^{n+1-} = (\alpha_1)_j^n - K_j^n \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* - u_{j-1/2}^* \right), \tag{38e}$$

where  $R_j$  is given by

$$R_{j} = 1 + \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^{*} - u_{j-1/2}^{*} \right). \tag{39}$$

Some properties of the numerical scheme, presented in section 3.6, employ these update formulae in the derivation.

#### 3.4. Numerical solution of the convective submodel

The convective system is approximated by using a classical upwind finite-volume scheme as employed in Chalons et al. [29]. Making again a forward Euler time step, the scheme reads:

$$\varphi_j^{n+1} = \varphi_j^{n+1-} - \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* \varphi_{j+1/2}^{n+1-} - u_{j-1/2}^* \varphi_{j-1/2}^{n+1-} \right) + \frac{\Delta t}{\Delta x} \varphi_j^{n+1-} \left( u_{j+1/2}^* - u_{j-1/2}^* \right).$$

$$(40)$$

where  $\varphi \in \{\rho, \rho u, \rho E, \rho Y_1, \alpha_1\}$ . The upwind value is used to approximate the interface value  $\varphi_{i+1/2}$ :

$$\varphi_{j+1/2}^{n+1-} = \begin{cases} \varphi_j^{n+1-}, & \text{if } u_{j+1/2}^* \ge 0, \\ \varphi_{j+1}^{n+1-}, & \text{if } u_{j+1/2}^* < 0. \end{cases}$$

$$(41)$$

## 3.5. Stability requirement

The common time step in the explicit time integration method is obtained using the Courant numbers of both subsystems. The Courant numbers are given by

$$C^a = \frac{\Delta t}{\Delta x} \max_j \lambda_j^a, \tag{42}$$

with maximum wave speed λ a <sup>j</sup> = max τ n j , τ <sup>n</sup> j+1 aj+1/2, for the acoustic subsystem, and by

$$C^c = \frac{\Delta t}{\Delta x} \max_j \lambda_j^c, \tag{43}$$

with the maximum wave speed λ c <sup>j</sup> = u ∗ j−1/2 + − u ∗ j+1/2 − , for the convective subsystem, where b <sup>±</sup> = (b±|b|)/2. The time step is determined by the requirement that both Courant numbers need to be less than one. In the implementation, the most severe time step restriction is taken for both subsystems. Hence, the time step size is selected with the Courant number C = max (C a , C c ). The Courant number for the classical direct approaches is defined by

$$C^{d} = \frac{\Delta t}{\Delta x} \max_{j} \left( |u_{j+1/2}| + c_{j+1/2} \right). \tag{44}$$

#### 3.6. Main properties scheme

#### 3.6.1. Conservation of mass, momentum, energy and partial mass

The scheme of the convective system [\(40\)](#page-7-3) can be written as:

$$\varphi_j^{n+1} = R_j \varphi_j^{n+1-} - \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* \varphi_{j+1/2}^{n+1-} - u_{j-1/2}^* \varphi_{j-1/2}^{n+1-} \right), \tag{45}$$

where R<sup>j</sup> is defined by [\(39\)](#page-7-4). Substitution of [\(38\)](#page-7-5) into this form leads to the update formulae

$$(\rho)_{j}^{n+1} = (\rho)_{j}^{n} - \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^{*} \rho_{j+1/2}^{n+1-} - u_{j-1/2}^{*} \rho_{j-1/2}^{n+1-} \right), \tag{46a}$$

$$(\rho u)_j^{n+1} = (\rho u)_j^n$$

$$-\frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* (\rho u)_{j+1/2}^{n+1-} + p_{j+1/2}^* - u_{j-1/2}^* (\rho u)_{j-1/2}^{n+1-} - p_{j-1/2}^* \right), \tag{46b}$$

$$(\rho E)_j^{n+1} = (\rho E)_j^n - \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* (\rho E)_{j+1/2}^{n+1-} + p_{j+1/2}^* u_{j+1/2}^* \right)$$

$$-u_{j-1/2}^*(\rho E)_{j-1/2}^{n+1-} - p_{j-1/2}^* u_{j-1/2}^* \right), \tag{46c}$$

$$(\rho Y_1)_j^{n+1} = (\rho Y_1)_j^n - \frac{\Delta t}{\Delta x} \left( u_{j+1/2}^* (\rho Y_1)_{j+1/2}^{n+1-} - u_{j-1/2}^* (\rho Y_1)_{j-1/2}^{n+1-} \right), \tag{46d}$$

which guarantees the conservation of mass, momentum, energy and partial mass of the proposed approach. Please notice that the choice of u ∗ j+1/2 in the transport scheme makes it possible to have a fully conservative scheme for the conservative variables [\[29\]](#page-28-16). Due to the non-conservative form of the topology equation, there is no conservation of the volume fraction.

# 3.6.2. Positivity of the volume fraction and mass fraction

Using the definition of the interfacial variable [\(11\)](#page-3-5), the update formula [\(38e\)](#page-7-6) of the volume fraction in the acoustic system can be written as

$$(\alpha_1)_j^{n+1-} = (\alpha_1)_j^n \left[ 1 - \frac{\Delta t}{\Delta x} (\alpha_2)_j^n \frac{(\rho_2 c_2^2)_j^n - (\rho_1 c_1^2)_j^n}{(\alpha_2)_j^n (\rho_1 c_1^2)_j^n + (\alpha_1)_j^n (\rho_2 c_2^2)_j^n} \left( u_{j-1/2}^* - u_{j+1/2}^* \right) \right]. \tag{47}$$

Since  $(\alpha_1)_j^n \geq 0$ , positivity of the volume fraction is ensured when the part within the brackets is positive, i.e.

$$A_j^n \frac{\Delta t}{\Delta x} \left( u_{j-1/2}^* - u_{j+1/2}^* \right) \le 1, \tag{48}$$

where

$$A_j^n = (\alpha_2)_j^n \frac{(\rho_2 c_2^2)_j^n - (\rho_1 c_1^2)_j^n}{(\alpha_2)_j^n (\rho_1 c_1^2)_j^n + (\alpha_1)_j^n (\rho_2 c_2^2)_j^n}.$$
(49)

The observations

$$\left(\rho_2 c_2^2\right)_j^n - \left(\rho_1 c_1^2\right)_j^n < \max\left[\left(\rho_1 c_1^2\right)_j^n, \left(\rho_2 c_2^2\right)_j^n\right],$$
 (50a)

$$(\alpha_2)_j^n \left(\rho_1 c_1^2\right)_j^n + (\alpha_1)_j^n \left(\rho_2 c_2^2\right)_j^n > \min\left[\left(\rho_1 c_1^2\right)_j^n, \left(\rho_2 c_2^2\right)_j^n\right], \tag{50b}$$

and  $0 \le (\alpha_2)_i^n \le 1$  imply that  $A_i^n \le 1$ . Using the CFL-type condition given in (44), we obtain

$$\frac{\Delta t}{\Delta x} \left( u_{j-1/2}^* - u_{j+1/2}^* \right) \le \frac{\Delta t}{\Delta x} \left[ \left( u_{j-1/2}^* \right)^+ - \left( u_{j+1/2}^* \right)^- \right] \le 1. \tag{51}$$

Positivity of the volume fraction is thus ensured by combining the results. Note that the upper bound  $(\alpha_1)_j^{n+1-} \leq 1$  is a direct consequence of this result. Similarly, the update formula (38d) ensures the positivity of the mass fraction.

#### 4. Numerical results

To illustrate the behavior of the proposed scheme, it is evaluated for five two-phase flow problems encountered in the literature: a translating interface problem, a pressure jump problem, a no-reflection problem, a water-air mixture problem and a two-phase cavitation problem. To illustrate the behavior of the proposed scheme, we consider standard shock-tube problems encountered in the literature.

All five test cases are defined such that no wave hits a boundary before the prescribed end time. All test cases are also computed using the direct HLLC-type approach proposed by Daude et al. [17]. The tests are performed with first-order accuracy in space and time. For each test, the Courant numbers of the current splitting approach and the direct approach are taken equal:  $\mathcal{C} = \mathcal{C}^d$ . The comparisons are performed using the same number of cells. The convergence rates are shown for each test case where an analytical solution is available. To compare the performance of both methods, the number of time steps and the CPU times are reported.

# 4.1. Translating two-phase interface

In this first test case, also considered in e.g. [21], a dense fluid and a much less dense gas move to the right, at constant velocity and pressure. The initial interface is located in the middle of the tube (x = 0.0) of length L = 0.5. This test case is considered to assess the behavior of the present scheme at a material interface with a density jump which is representative for that of the important class of water-air flows.

Table 1: Initial values and material properties for the translating interface problem.

|         | (a) Initial va | (b) Materi | al properties |       |            |         |          |
|---------|----------------|------------|---------------|-------|------------|---------|----------|
|         | ρ              | u          | p             | $Y_1$ | $\alpha_1$ |         | $\gamma$ |
| Fluid 1 | 1000           | 1.0        | 1.0           | 1.0   | 1.0        | Fluid 1 | 1.4      |
| Fluid 2 | 1.0            | 1.0        | 1.0           | 0.0   | 0.0        | Fluid 2 | 1.6      |

The initial values and material properties are given in Table [1.](#page-9-1) Two perfect gases are considered (π<sup>1</sup> = π<sup>2</sup> = 0, η<sup>1</sup> = η<sup>2</sup> = 0), with the difference for both fluids only in γ. The depicted results have been obtained at time t = 0.1 with N = 400 cells and a Courant number C = 0.95. The distributions of the primitive variables are visualized in the Figures [2](#page-10-0) to [5](#page-11-0) and the convergence rates of the density profiles are listed in Table [2.](#page-12-0)

![](_page_10_Figure_1.jpeg)

Figure 2: Translating interface problem - density profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.1.

![](_page_10_Figure_3.jpeg)

Figure 3: Translating interface problem - velocity profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.1.

![](_page_11_Figure_0.jpeg)

Figure 4: Translating interface problem - pressure profile - Exact solution "-", splitting approach "o" and direct approach "+" at t = 0.1.

![](_page_11_Figure_2.jpeg)

Figure 5: Translating interface problem - volume and mass fraction profiles - Exact solution "-", splitting approach "o" and direct approach "+" at t=0.1.

The results obtained with the proposed splitting-based method are very similar to the ones obtained with the direct approach from [17]. The contact discontinuity is well retrieved with both methods, whereas the velocity and pressure profiles are perfectly constant; no pressure oscillations occur across the interface. The location of the two-phase interface for the mass fraction is a bit off (see Figure 6), for both the proposed method and the direct approach from [17]. This is also the case for the method proposed in [21]. In the region where the material interface is smeared due to intrinsic numerical dissipation of the two numerical schemes, the associated cells contain both fluids with  $\alpha_2 \rho_2 \ll \alpha_1 \rho_1$  which gives a value of  $Y_1$  close to 1. With a finer mesh, the correct location is obtained, see also Table 2. At the end time t = 0.1 the contact discontinuity is indeed located at x = 0.1. The proposed method captures the location slightly better. The newly proposed method takes larger time steps (124 time steps) than the direct approach from [17] (192 time

![](_page_12_Figure_0.jpeg)

Figure 6: Translating interface problem - zoom at contact discontinuity - Exact solution "-", splitting approach "o" and direct approach "+" at t = 0.1.

Table 2: The  $L_1$ -convergence rates for the density of the translating interface problem. The convergence rates are computed as  $c_N = \log(e_N/e_{2N})/\log(2)$ . The errors are given by  $e_N = \|s_N - s_{\mathrm{exact}}\|_{L_1}$ , where  $s_N$  is the solution computed with N grid points,  $s_{\mathrm{exact}}$  the exact solution, and  $\|\cdot\|_{L_1}$  the standard  $L_1$ -norm.

| Convergence rates | Splitting | Direct |
|-------------------|-----------|--------|
| $c_{40}$          | 0.67      | 0.56   |
| $c_{80}$          | 0.64      | 0.53   |
| $c_{160}$         | 0.63      | 0.52   |
| c <sub>320</sub>  | 0.60      | 0.51   |
| $c_{640}$         | 0.57      | 0.50   |

steps). The CPU time is 0.17s and 0.36s for the splitting approach and the direct approach, respectively (averaged over 500 runs on an i5 processor). Both methods show similar convergence rates, see Table 2.

## 4.2. A two-pressure jump problem

In this test case, proposed by Barberon al. [\[42\]](#page-29-7) and also considered in [\[17\]](#page-28-10), the shock tube is again filled with two perfect gases with different densities. The pressures at both sides are slightly different. The interface is located at x = 0.5 m. Due to the pressure difference, a shock wave will propagate rightwards and a rarefaction wave will propagate leftwards.

Table 3: Initial values and material properties for the two-pressure jump problem. The dimensions of the quantities ρ, u and p are kg m−<sup>3</sup> , m s−<sup>1</sup> and Pa, respectively.

|         | (a) Initial values |      | (b) Material properties |     |     |                |
|---------|--------------------|------|-------------------------|-----|-----|----------------|
|         | ρ                  | u    | p                       | Y1  | α1  | γ              |
| Fluid 1 | 10                 | 50.0 | 1.1 · 105               | 1.0 | 1.0 | Fluid 1<br>1.4 |
| Fluid 2 | 1.0                | 50.0 | 1.0 · 105               | 0.0 | 0.0 | Fluid 2<br>1.1 |

The initial values and material properties are given in Table [3.](#page-13-0) Also here the SG EOS reduces to the PG EOS. The results are obtained at time t = 1.0 ms with N = 400 cells for the Courant number of C = 0.95. The distributions of the primitive variables at t = 1.0 ms are depicted in Figures [7-](#page-13-1)[11](#page-16-0) and the convergence rates are listed in Table [4.](#page-15-0)

![](_page_13_Figure_5.jpeg)

Figure 7: Two-pressure jump problem - density profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 1.0 ms.

Again, the results obtained with the proposed method are very similar to the ones obtained with the unsplit approach from [\[17\]](#page-28-10). The location of the shock wave is accurately captured with both methods, also in the zoom (Figure [11\)](#page-16-0) no significant difference is visible. Also for this test case, the newly proposed method takes larger time steps (149 time steps) than the direct approach from [\[17\]](#page-28-10) (166 time steps). The CPU time is 0.26s and 0.69s for the splitting approach and the direct approach respectively (averaged over 500 runs on an i5 processor). Again, both methods show similar convergence rates, see Table [4.](#page-15-0)

![](_page_14_Figure_0.jpeg)

Figure 8: Two-pressure jump problem - velocity profileExact solution "-", splitting approach "◦" and direct approach "+"at t = 1.0 ms.

![](_page_14_Figure_2.jpeg)

Figure 9: Two-pressure jump problem - pressure profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 1.0 ms.

![](_page_15_Figure_0.jpeg)

Figure 10: Two-pressure jump problem - mass and volume fraction profile - Exact solution "-", splitting approach "o" and direct approach "+" at t = 1.0 ms.

Table 4: The  $L_1$ -convergence rates for the two-pressure jump problem. The convergence rates are computed as  $c_N = \log(e_N/e_{2N})/\log(2)$ . The errors are given by  $e_N = \|s_N - s_{\text{exact}}\|_{L_1}$ , where  $s_N$  is the solution computed with N grid points,  $s_{\text{exact}}$  the exact solution, and  $\|\cdot\|_{L_1}$  the standard  $L_1$ -norm.

| Convergence rates |           |      | Physical quantity |      |       |            |  |  |
|-------------------|-----------|------|-------------------|------|-------|------------|--|--|
| Splitting         | approach  | ρ    | u                 | p    | $Y_1$ | $\alpha_1$ |  |  |
|                   | $c_{40}$  | 0.43 | 0.69              | 0.65 | 0.88  | 0.41       |  |  |
|                   | $c_{80}$  | 0.54 | 0.59              | 0.49 | 0.31  | 0.56       |  |  |
|                   | $c_{160}$ | 0.50 | 0.65              | 0.59 | 0.51  | 0.50       |  |  |
|                   | $c_{320}$ | 0.50 | 0.58              | 0.54 | 0.50  | 0.50       |  |  |
|                   | $c_{640}$ | 0.50 | 0.59              | 0.56 | 0.50  | 0.50       |  |  |
| Direct            | approach  | ρ    | u                 | p    | $Y_1$ | $\alpha_1$ |  |  |
|                   | $c_{40}$  | 0.42 | 0.70              | 0.69 | 0.86  | 0.40       |  |  |
|                   | $c_{80}$  | 0.54 | 0.56              | 0.46 | 0.30  | 0.56       |  |  |
|                   | $c_{160}$ | 0.50 | 0.69              | 0.61 | 0.51  | 0.49       |  |  |
|                   | $c_{320}$ | 0.50 | 0.59              | 0.54 | 0.50  | 0.50       |  |  |
|                   | $c_{640}$ | 0.50 | 0.62              | 0.58 | 0.50  | 0.50       |  |  |

![](_page_16_Figure_0.jpeg)

Figure 11: Two-pressure jump problem - zoom - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 1.0 ms.

## 4.3. No-reflection problem

The third test we perform is the so-called no-reflection problem, which is also assessed in [\[21\]](#page-28-8). In this test case, the right state is initially at rest and the left state moves towards the right state. The density and pressure of the left state are high compared with the right state. This will cause the two-fluid interface and the shock wave to move rightwards. The initial conditions of the perfect gases are chosen such that no reflection wave occurs.

|  |  | Table 5: Initial values and material properties for the no-reflection problem. |  |
|--|--|--------------------------------------------------------------------------------|--|
|  |  |                                                                                |  |

|         | (a) Initial values |        |     | (b) Material properties |     |         |       |
|---------|--------------------|--------|-----|-------------------------|-----|---------|-------|
|         | ρ                  | u      | p   | Y1                      | α1  |         | γ     |
| Fluid 1 | 3.1748             | 9.4350 | 100 | 1.0                     | 1.0 | Fluid 1 | 1.667 |
| Fluid 2 | 1.0                | 0.0    | 1.0 | 0.0                     | 0.0 | Fluid 2 | 1.2   |

The initial values and material properties are given in Table [5.](#page-17-0) The results are obtained at time t = 0.02 with N = 400 cells with a CFL number of C = 0.95. The results are visualized in the Figures [12-](#page-17-1)[17](#page-21-0) and the convergence rates are listed in Table [6.](#page-19-0)

![](_page_17_Figure_5.jpeg)

Figure 12: No-reflection problem - density profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.02.

The location of the contact discontinuity is satisfactorily retrieved with both methods. A small reflected wave is visible at around x = 0.05, which is weaker for the splitting-based scheme (see Figure [16\)](#page-20-0). For both methods it vanishes when refining the grid. The shock wave is well retrieved with both methods. The newly proposed method seems to be less diffusive than the direct approach (see Figure [17\)](#page-21-0). Again, the newly proposed method takes larger time steps (169 time steps) than the direct approach from [\[17\]](#page-28-10) (285 time steps). The CPU time is 0.25s and 0.33s for the splitting approach and the direct approach, respectively (averaged over 500 runs on an i5 processor).

![](_page_18_Figure_0.jpeg)

Figure 13: No-reflection problem - velocity profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.02.

![](_page_18_Figure_2.jpeg)

Figure 14: No-reflection problem - pressure profile - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.02.

![](_page_19_Figure_0.jpeg)

Figure 15: No-reflection problem - mass and volume fraction profile - Exact solution "-", splitting approach "o" and direct approach "+" at t=0.02.

Table 6: The  $L_1$ -convergence rates for the no-reflection problem. The errors are computed as  $e_N = \|s_N - s_{\text{exact}}\|_{L_1}$ , where  $s_N$  is the solution computed with N grid points,  $s_{\text{exact}}$  the exact solution, and  $\|\cdot\|_{L_1}$  the standard  $L_1$ -norm.

| Fra       | action errors |      | Physical quantity |      |       |            |  |
|-----------|---------------|------|-------------------|------|-------|------------|--|
| Splitting | approach      | ρ    | u                 | p    | $Y_1$ | $\alpha_1$ |  |
|           | $c_{40}$      | 0.69 | 1.05              | 1.04 | 0.50  | 0.45       |  |
|           | $c_{80}$      | 0.82 | 1.33              | 1.22 | 0.52  | 0.46       |  |
|           | $c_{160}$     | 0.57 | 0.83              | 0.90 | 0.52  | 0.45       |  |
|           | $c_{320}$     | 0.57 | 0.80              | 0.82 | 0.43  | 0.50       |  |
|           | $c_{640}$     | 0.68 | 1.27              | 1.22 | 0.43  | 0.50       |  |
| Direct    | approach      | ρ    | u                 | p    | $Y_1$ | $\alpha_1$ |  |
|           | $c_{40}$      | 0.44 | 1.03              | 0.86 | 0.42  | 0.42       |  |
|           | $c_{80}$      | 0.54 | 1.17              | 0.93 | 0.42  | 0.38       |  |
|           | $c_{160}$     | 0.50 | 0.87              | 0.90 | 0.45  | 0.39       |  |
|           | $c_{320}$     | 0.51 | 0.81              | 0.91 | 0.41  | 0.42       |  |
|           | $c_{640}$     | 0.56 | 1.20              | 1.06 | 0.44  | 0.45       |  |

![](_page_20_Figure_0.jpeg)

Figure 16: No-reflection problem - zoom at bumps - Exact solution "-", splitting approach "◦" and direct approach "+"at t = 0.02.

![](_page_21_Figure_0.jpeg)

Figure 17: No-reflection problem - zoom at shock wave - Exact solution "-", splitting approach "o" and direct approach "+" at t = 0.02.

## 4.4. Water-air mixture problem

In this shock tube test we consider a water-air mixture problem. This test case has been considered by Murrone and Guillard [20] and by Kreeft and Koren [21]. In contrast to the previous test cases, the shock tube is now filled with a mixture of water and air  $(0 < Y_1, \alpha_1 < 1)$  and stiffened gases are considered. Both mixture states are initially at rest and the initial pressure ratio is  $10^4$ .

Table 7: Initial values and material properties for the water-air mixture problem. The dimensions of the quantities  $\rho$ , u and p are kg m<sup>-3</sup>, m s<sup>-1</sup> and Pa respectively.

|               | ρ   | u   | p        | $Y_1$  | $\alpha_1$ |
|---------------|-----|-----|----------|--------|------------|
| Left chamber  | 525 | 0.0 | $10^{9}$ | 0.0476 | 0.5        |
| Right chamber | 525 | 0.0 | $10^{5}$ | 0.9524 | 0.5        |

Table 8: Material properties for the water-air mixture problem. The dimensions of the quantities  $\pi$  and  $\eta$  are Pa and J kg<sup>-1</sup> respectively.

|         | $\gamma$ | $\pi$            | $\eta$ |  |
|---------|----------|------------------|--------|--|
| Fluid 1 | 1.4      | 0.0              | 0.0    |  |
| Fluid 2 | 4.4      | $6 \cdot 10^{8}$ | 0.0    |  |

The initial values and material properties are given in Tables 7 and 8. Numerical results are obtained at time  $t = 200 \ \mu s$  with N = 400 cells with CFL number C = 0.95. The results are visualized in the Figures 18-22.

![](_page_22_Figure_7.jpeg)

Figure 18: Water-air mixture problem - Density profile - Numerical solution from [20] "x", splitting approach "o" and direct approach "+" at  $t = 200\mu$  s.

The numerical results are in good agreement with numerical solutions from Murrone and Guillard [20]. The volume fraction distribution on the right side of the middle wave shows slightly different values for all

![](_page_23_Figure_0.jpeg)

Figure 19: Water-air mixture problem - Velocity profile - Numerical solution from [\[20\]](#page-28-7) "x", splitting approach "◦" and direct approach "+"at t = 200µ s.

![](_page_23_Figure_2.jpeg)

Figure 20: Water-air mixture problem - Pressure profile - Numerical solution from [\[20\]](#page-28-7) "x", splitting approach "◦" and direct approach "+"at t = 200µ s.

three schemes. The numerical solution from [\[20\]](#page-28-7) shows a slightly lower value compared with the splittingbased method and a slightly higher value than the HLLC-type scheme. This test case indicates that the proposed method can also deal with mixture problems. Also for this test case, the newly proposed method takes larger time steps (179 time steps) than the direct approach from [\[17\]](#page-28-10) (giving 193 time steps). The CPU time is 0.25s and 0.38s for the splitting approach and the direct approach, respectively (averaged over 500 runs on an i5 processor).

![](_page_24_Figure_0.jpeg)

Figure 21: Water-air mixture problem - Volume fraction profile - Numerical solution from [20] "x", splitting approach " $\circ$ " and direct approach "+" at  $t=200\mu$  s.

![](_page_24_Figure_2.jpeg)

Figure 22: Water-air mixture problem - zoom - Numerical solution from [20] "x", splitting approach "o" and direct approach "+" at  $t=200\mu$  s.

#### 4.5. Two-phase cavitation problem

In this test case proposed by Saurel et al. [19] the tube is filled with water and its vapor at atmospheric pressure. Thus a mixture of the fluids is considered: initially the water (with density  $\rho_2 = 1150 \text{ kg m}^{-3}$ ) contains a small portion of vapor  $\alpha_1 = 10^{-2}$  (with density  $\rho_1 = 0.63 \text{ kg m}^{-3}$ ). An initial velocity discontinuity separates both states.

Table 9: Initial values for the two-phase cavitation problem. The dimensions of the quantities  $\rho$ , u and p respectively.

|               | $\rho$    | u    | p        | $Y_1$                   | $\alpha_1$ |
|---------------|-----------|------|----------|-------------------------|------------|
| Left chamber  | 1138.5063 | -2.0 | $10^{5}$ | $5.53356 \cdot 10^{-6}$ | 0.01       |
| Right chamber | 1138.5063 | 2.0  | $10^{5}$ | $5.53356 \cdot 10^{-6}$ | 0.01       |

Table 10: Material properties for the two-phase cavitation problem. The dimensions of the quantities  $\pi$  and  $\eta$  are Pa and J kg<sup>-1</sup> respectively.

|         | $\gamma$ | $\pi$    | η                   |  |
|---------|----------|----------|---------------------|--|
| Fluid 1 | 2.35     | $10^{9}$ | $-1167 \cdot 10^3$  |  |
| Fluid 2 | 1.43     | 0        | $2030 \cdot 10^{3}$ |  |

The initial values and material properties are given in Tables 9 and 10. Numerical results are presented at time t = 3.2 ms with N = 400 cells. A smaller time step (CFL = C = 0.01) is used due to the strong rarefaction wave. The results are visualized in the Figures 23-26.

![](_page_25_Figure_7.jpeg)

Figure 23: Two-phase cavitation problem problem - Density profile - Splitting approach "o" and direct approach "+" at t = 3.2 ms.

Both methods give very similar results, consistent with those obtained in [19, 43–45]. The density and volume fraction profiles obtained with the splitting approach show some overshooting in the middle

![](_page_26_Figure_0.jpeg)

Figure 24: Two-phase cavitation problem problem - Velocity profile - Splitting approach "◦" and direct approach "+"at t = 3.2 ms.

![](_page_26_Figure_2.jpeg)

Figure 25: Two-phase cavitation problem problem - Pressure profile - Splitting approach "◦" and direct approach "+"at t = 3.2 ms.

region. This test case indicates that a strong rarefaction wave is well retrieved with both methods. Again, the newly proposed method takes larger time steps (14303 time steps) than the direct approach from [\[17\]](#page-28-10) (14559 time steps). The CPU time is 23.5s and 27.4s for the splitting approach and the direct approach,

![](_page_27_Figure_0.jpeg)

Figure 26: Two-phase cavitation problem problem - Mass and volume fraction profile - Splitting approach " $\circ$ " and direct approach "+" at t = 3.2 ms.

respectively (averaged over 10 runs on an i5 processor).

#### 5. Conclusions

An acoustic-convective splitting-based scheme has been proposed to solve the Kapila single-pressure single-velocity two-phase flow model. The acoustic and convective submodels are alternatingly stepped in time to approximate the solution of the entire flow model. The model dealing with the acoustic waves has been cast into a Lagrangian form, and solved using an HLLC-type solver. This approach gives a simple numerical scheme. The model dealing with the convective waves has been approximated using a classical upwind scheme. The method has been evaluated for a variety of shock tube problems, and compared with an existing HLLC-type scheme applied to the original (unsplit) Kapila model. The obtained numerical results demonstrate the ability of the proposed method to deal with strong discontinuities and mixture flows. They are in good agreement with exact and approximate reference solutions. The newly proposed method takes larger time steps than the HLLC-type scheme does for the unsplit model originally proposed in [17]. This is most significant in the transonic regime. Contact discontinuities, rarefaction waves and shock waves are captured very accurately with both the new method and the direct approach. The new method seems to be less diffusive than the direct approach. Furthermore, the splitting approach may circumvent the inaccuracies when using approximate Godunov approaches for subsonic flows. The potential of the current method to deal with low-Mach number flows is briefly described in [40]. To obtain higher-order temporal accuracy, the combination of higher-order methods to solve the systems together with a higher-order splitting approach must be used. One approach could be to use a generalized- $\alpha$  or a Runge-Kutta time integrator combined with a Strang splitting approach. The proposed approach has a natural extension to multi-dimensional problems.

#### References

[1] M.R. Baer and J.W. Nunziato. A two-phase mixture theory for the deflagration-to-detonation transition (DDT) in reactive granular materials. *International Journal of Multiphase Flow*, 12:861–889, 1986.

- [2] E. Romenski, A.D. Resnyansky, and E.F. Toro. Conservative hyperbolic formulation for compressible two-phase flow with different phase pressures and temperatures. Quarterly of Applied Mathematics, 65:259–279, 2007.
- [3] F. Crouzet, F. Daude, P. Galon, P. Helluy, J.-M. H´erard, O. Hurisse, and Y. Liu. Approximate solutions of the Baer-Nunziato model. In ESAIM: Proceedings, volume 40, pages 63–82. EDP Sciences, 2013.
- [4] R. Saurel and R. Abgrall. A multiphase Godunov method for compressible multifluid and multiphase flows. Journal of Computational Physics, 150:425–467, 1999.
- [5] J. Massoni, R. Saurel, B. Nkonga, and R. Abgrall. Some models and Eulerian methods for interface problems between compressible fluids with heat transfer. International Journal of Heat and Mass Transfer, 45:1287–1307, 2002.
- [6] D.W. Schwendeman, C.W. Wahle, and A.K. Kapila. The Riemann problem and a high-resolution Godunov method for a model of compressible two-phase flow. Journal of Computational Physics, 212:490–526, 2006.
- [7] T. Gallou¨et, J.-M. H´erard, and N. Seguin. Numerical modeling of two-phase flows using the two-fluid two-pressure approach. Mathematical Models and Methods in Applied Sciences, 14:663–700, 2004.
- [8] S.A. Tokareva and E.F. Toro. HLLC-type Riemann solver for the Baer–Nunziato equations of compressible two-phase flow. Journal of Computational Physics, 229:3573–3604, 2010.
- [9] M. Dumbser and E.F. Toro. A simple extension of the Osher Riemann solver to non-conservative hyperbolic systems. Journal of Scientific Computing, 48:70–88, 2011.
- [10] A. Ambroso, C. Chalons, and P-A Raviart. A Godunov-type method for the seven-equation model of compressible two-phase flow. Computers & Fluids, 54:67–91, 2012.
- [11] J.-M. H´erard and O. Hurisse. A fractional step method to compute a class of compressible gas–liquid flows. Computers & Fluids, 55:57–69, 2012.
- [12] F. Crouzet, F. Daude, P. Galon, J.-M. H´erard, O. Hurisse, and Y. Liu. Validation of a two-fluid model on unsteady liquid–vapor water flows. Computers & Fluids, 119:131–142, 2015.
- [13] H. Lochon, F. Daude, P. Galon, and J.-M. H´erard. Comparison of two-fluid models on steam-water transients. ESAIM: Mathematical Modelling and Numerical Analysis (2016), available online, 2016.
- [14] F. Daude and P. Galon. On the computation of the Baer–Nunziato model using ALE formulation with HLL-and HLLCtype solvers towards fluid–structure interactions. Journal of Computational Physics, 304:189–230, 2016.
- [15] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, and D.S. Stewart. Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations. Physics of Fluids, 13:3002–3024, 2001.
- [16] G. Allaire, S. Clerc, and S. Kokh. A five-equation model for the simulation of interfaces between compressible fluids. Journal of Computational Physics, 181:577–616, 2002.
- [17] F. Daude, P. Galon, Z. Gao, and E. Blaud. Numerical experiments using a HLLC-type scheme with ALE formulation for compressible two-phase flows five-equation models with phase transition. Computers & Fluids, 94:112–138, 2014.
- [18] S. Kokh and F. Lagouti`ere. An anti-diffusive numerical scheme for the simulation of interfaces between compressible fluids by means of a five-equation model. Journal of Computational Physics, 229:2773–2809, 2010.
- [19] R. Saurel, F. Petitpas, and R. Abgrall. Modelling phase transition in metastable liquids: application to cavitating and flashing flows. Journal of Fluid Mechanics, 607:313–350, 2008.
- [20] A. Murrone and H. Guillard. A five equation reduced model for compressible two phase flow problems. Journal of
- Computational Physics, 202:664–698, 2005. [21] J.J. Kreeft and B. Koren. A new formulation of Kapila's five-equation model for compressible two-fluid flow, and its numerical treatment. Journal of Computational Physics, 229:6220–6242, 2010.
- [22] M. Ahmed, M.R. Saleem, S. Zia, and S. Qamar. Central upwind scheme for a compressible two-phase flow model. PloS ONE, 10:e0126273 1–26, 2015.
- [23] R.A. Berry, R. Saurel, and F. Petitpas. A simple and efficient diffuse interface method for compressible two–phase flows. In International Conference on Advances on Mathematics, Computational Methods and Reactor Physics (M&C 2009), New York, 2009.
- [24] F. Petitpas, E. Franquet, R. Saurel, and O. Le Metayer. A relaxation-projection method for compressible flows. Part II: Artificial heat exchanges for multiphase shocks. Journal of Computational Physics, 225:2214–2248, 2007.
- [25] R. Saurel, E. Franquet, E. Daniel, and O. Le Metayer. A relaxation-projection method for compressible flows. Part I: The numerical equation of state for the Euler equations. Journal of Computational Physics, 223:822–845, 2007.
- [26] R. Abgrall and V. Perrier. Asymptotic expansion of a multiscale numerical scheme for compressible multiphase flow. Multiscale Modeling & Simulation, 5:84–115, 2006.
- [27] R. Saurel, F. Petitpas, and R.A. Berry. Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures. Journal of Computational Physics, 228:1678–1712, 2009.
- [28] J. Jiang, Y. Fu, L. Zhang, Y. Li, W. Ji, and Y. Liu. The investigation of gas–liquid two-phase transient flow based on steger–warming flux vector splitting method in pipelines. Advances in Mechanical Engineering, 8:1–11, 2016.
- [29] C. Chalons, M. Girardin, and S. Kokh. An all-regime Lagrange-Projection like scheme for the gas dynamics equations on unstructured meshes. Communications in Computational Physics, 20:188–233, 2016.
- [30] M.F.P. ten Eikelder, F. Daude, and B. Koren. A Lagrange-Projection-like numerical scheme for mixed acoustic-convective two-phase flows. Proceedings of the ASME 2016 Pressure Vessels & Piping Conference, Vancouver, Canada, July 2016, Paper PVP2016-63539.
- [31] G. Huber, S. Tanguy, J.-C. B´era, and B. Gilles. A time splitting projection scheme for compressible two-phase flows. Application to the interaction of bubbles with ultrasound waves. Journal of Computational Physics, 302:439–468, 2015.
- [32] T. Fl˚atten, A. Morin, and S.T. Munkejord. On solutions to equilibrium problems for systems of stiffened gases. SIAM Journal on Applied Mathematics, 71:41–67, 2011.
- [33] R. Saurel and R. Abgrall. A simple method for compressible multifluid flows. SIAM Journal on Scientific Computing,

- 21:1115–1145, 1999.
- [34] V. Coralic and T. Colonius. Shock-induced collapse of a bubble inside a deformable vessel. European Journal of Mechanics-B/Fluids, 40:64–74, 2013.
- [35] A. B. Gojani, K. Ohtani, K. Takayama, and S. H. R. Hosseini. Shock Hugoniot and equations of states of water, castor oil, and aqueous solutions of sodium chloride, sucrose and gelatin. Shock Waves, 26:63–68, 2016.
- [36] R. Abgrall and R. Saurel. Discrete equations for physical and numerical compressible multiphase mixtures. Journal of Computational Physics, 186:361–396, 2003.
- [37] H. Lund and P. Aursand. Splitting methods for relaxation two-phase flow models. International Journal of Materials Engineering Innovation, 4:117–131, 2013.
- [38] A.B. Wood. A Textbook of Sound. 1930. G. Bell and Sons Ltd, 1930.
- [39] R.J. LeVeque. Finite Volume Methods for Hyperbolic Problems. Cambridge University Press, 2002.
- [40] M.F.P. ten Eikelder. Compressible five-equation two-phase flow models towards the computation of the water hammer phenomenon. Master's thesis, Eindhoven University of Technology, the Netherlands (2015), available from [http:](http://alexandria.tue.nl/extra1/afstversl/wsk-i/Eikelder_2015.pdf) [//alexandria.tue.nl/extra1/afstversl/wsk-i/Eikelder\\_2015.pdf](http://alexandria.tue.nl/extra1/afstversl/wsk-i/Eikelder_2015.pdf).
- [41] E.F. Toro, M. Spruce, and W. Speares. Restoration of the contact surface in the HLL-Riemann solver. Shock Waves, 4:25–34, 1994.
- [42] T. Barberon, P. Helluy, and S. Rouy. Practical computation of axisymmetrical multifluid flows. International Journal of Finite Volumes, 1:1–34, 2003.
- [43] M.G. Rodio and R. Abgrall. An innovative phase transition modeling for reproducing cavitation through a five-equation model and theoretical generalization to six and seven-equation models. International Journal of Heat and Mass Transfer, 89:1386–1401, 2015.
- [44] M. Pelanti and K.M. Shyue. A mixture-energy-consistent six-equation two-phase numerical model for fluids with interfaces, cavitation and evaporation waves. Journal of Computational Physics, 259:331–357, 2014.
- [45] A. Zein, M. Hantke, and G. Warnecke. Modeling phase transition for compressible two-phase flows applied to metastable liquids. Journal of Computational Physics, 229:2964–2998, 2010.