# PRESSURE-EQUILIBRIUM-PRESERVING AND FULLY CONSERVATIVE DISCRETIZATION OF COMPRESSIBLE FLOW EQUATIONS FOR REAL AND THERMALLY PERFECT GASES

# A PREPRINT

# [Gennaro Coppola](https://orcid.org/0000-0003-4943-9551)

Dipartimento di Ingegneria Industriale Università di Napoli "Federico II" Napoli, Italy gcoppola@unina.it

## [Alessandro Aiello](https://orcid.org/0009-0003-6376-768X)

Dipartimento di Ingegneria Industriale Università di Napoli "Federico II" Napoli, Italy alessandro.aiello@unina.it

# [Carlo De Michele](https://orcid.org/0000-0002-6518-3114)

Gran Sasso Science Institute (GSSI) L'Aquila, Italy carlo.demichele@gssi.it

May 4, 2026

# ABSTRACT

Numerical simulations of compressible real-fluid flows are notoriously plagued by spurious pressure oscillations arising in regions of abrupt flow variations. As a possible remedy, several numerical formulations enforce the pressure equilibrium condition for the compressible Euler equations, typically at the cost of spoiling the correct conservation of total energy or by overspecifying the thermodynamical variables. This study proposes for the first time a numerical discretization procedure which is able to discretely preserve the full conservation of the linear invariants (mass, momentum and total energy) and to exactly enforce the pressure equilibrium condition. The method also preserves the conservation of kinetic energy by convection, and is based on the specification of nonlinear numerical fluxes for mass and internal energy which depend on the details of the equation of state. Both thermally perfect and real gases with an arbitrary equation of state are considered, and a simplified approximate pressure equilibrium preserving formulation with excellent performances is also proposed. The effectiveness of the novel formulations is assessed through a series of numerical simulations in supercritical and transcritical conditions with some of the most popular cubic equations of state.

*Keywords* Compressible flow · Pressure-equilibrium-preserving methods · Kinetic-energy-preserving methods · Real gases · Thermally perfect gases

# 1 Introduction

The accurate numerical simulation of compressible flows for real and thermally perfect gases is of great importance in a wide array of modern engineering and scientific disciplines. From the design of high-speed aerospace vehicles and advanced propulsion systems to the analysis of supercritical fluids in power generation and high-pressure turbomachinery, the assumption of a calorically perfect gas frequently breaks down [\[1,](#page-18-0) [2,](#page-18-1) [3\]](#page-18-2). In these extreme thermodynamic regimes, the fluid behavior deviates significantly from the ideal gas law, requiring the use of complex, non-linear Equations of State (EoS) to model phenomena such as variable specific heats, dense gas effects, and phase transitions.

However, the robust discretization of the compressible Euler equations coupled with a generic real gas EoS presents profound numerical challenges. Foremost among these is the generation of spurious, non-physical oscillations across contact discontinuities [\[4](#page-18-3)]. This phenomenon is fundamentally analogous to the well-documented difficulties encountered in the simulation of multicomponent flows [\[5\]](#page-18-4). Just as the abrupt variation of the specific heat ratio across a multi-fluid interface triggers severe numerical artifacts, the highly non-linear dependence of pressure on density and internal energy in a real gas EoS induces similar unphysical fluctuations. If left unmitigated, these spurious oscillations in pressure and velocity can propagate rapidly, leading to non-physical thermodynamic states and ultimately resulting in the catastrophic failure of the simulation.

The root of this issue lies in the extreme difficulty of simultaneously satisfying two fundamental numerical requirements: preserving local pressure equilibrium and maintaining strict, discrete conservation of total energy. When fully conservative numerical schemes are applied to real gases, the primary variables (density, momentum and total energy) are updated and exactly conserved. However, the subsequent non-linear inversion of the EoS, required to recover the pressure from the updated density and internal energy, frequently fails to maintain pressure equilibrium across moving contact discontinuities. Conversely, many existing strategies attempt to enforce a Pressure-Equilibrium-Preserving (PEP) condition by modifying the energy equation or employing primitive-variable formulations (i.e., discretizing the pressure equation directly in place of the total energy one). While these successfully suppress spurious oscillations, they inherently sacrifice exact discrete conservation of total energy. This loss of full conservation can be highly detrimental, as, in the presence of strong shocks, the scheme may fail to converge to the correct weak solution [\[5,](#page-18-4) [6\]](#page-18-5).

To address the spurious oscillations, a variety of numerical strategies have been proposed over the past decades. Early efforts, predominantly rooted in the study of multicomponent flows, focused on quasi-conservative formulations. As demonstrated by Karni [\[7](#page-18-6)], discretizing the Euler equations in primitive variables avoids the energy-to-pressure EoS inversion, successfully maintaining local pressure equilibrium. To recover partial conservation, Abgrall [\[5\]](#page-18-4) and later Shyue [\[8\]](#page-18-7) introduced quasi-conservative schemes wherein the standard conservative fluid equations are solved alongside non-conservative advection equations for the thermodynamic parameters (e.g., the specific heat ratio). By carefully discretizing these auxiliary equations, these methods achieve the PEP property. However, the explicit inclusion of non-conservative terms inherently sacrifices exact discrete conservation of total energy, leading to incorrect shock propagation speeds and thermodynamic anomalies across strong discontinuities. Similar problems were encountered by evolving the pressure equation for transcritical and supercritical fluids [\[4,](#page-18-3) [9\]](#page-18-8), and even the use of enthalpy and internal energy has been attempted [\[10\]](#page-18-9).

Recognizing the deficiencies of globally non-conservativemodels, double-flux methods have been introduced to bridge the gap between interface stability and shock fidelity. Originally formalized by Abgrall and Karni [\[11](#page-18-10)], this approach was later adapted for real-gas and transcritical flows by Ma et al. [\[12\]](#page-18-11). The double-flux strategy employs standard, fully conservative fluxes in the bulk of the flow, but locally switches to a thermodynamically "frozen", non-conservative auxiliary flux at sharp density gradients to enforce pressure equilibrium. While highly effective at mitigating unphysical waves in practice, double-flux methods still rely on local non-conservation. Furthermore, they require complex and empirical sensor functions to blend the two flux formulations, making them computationally rigid and highly sensitive to tuning parameters when applied to arbitrary, highly non-linear real gases.

To overcome these limitations, efforts were made to modify numerical fluxes directly within a fully conservative framework. For multicomponent flows, Fujiwara et al. [\[13](#page-18-12)] sought to address this by deriving equilibrium conditions to construct specialized numerical fluxes without relying on auxiliary transport equations. Terashima et al. [\[14\]](#page-18-13) extended this approach to the complex thermodynamic regimes of real gases, developing consistent numerical fluxes maintaining exact conservation of the primary variables. However, the final formulation yields only an approximately PEP scheme, with a fixed spatial order of accuracy. As a result, residual pressure oscillations can still manifest when simulating extremely severe thermodynamic gradients or phase transitions.

An alternative philosophy has sought to guarantee exact pressure equilibrium and numerical robustness by coupling PEP conditions with Kinetic-Energy-Preserving (KEP) schemes. The KEP framework was originally developed to prevent non-linear aliasing errors in standard compressible turbulence simulations without relying on numerical dissipation [\[15](#page-18-14), [16](#page-18-15), [17](#page-18-16)]. For calorically perfect ideal gases, researchers have successfully derived discretizations that combine KEP properties with exact pressure equilibrium [\[18,](#page-18-17) [19,](#page-18-18) [20](#page-18-19), [21,](#page-18-20) [22\]](#page-18-21). For real-fluid simulations, a recent work by Bernades et al. [\[23](#page-18-22)] proposed a specifically designed KEP and PEP scheme. Yet, to strictly maintain both pressure equilibrium and kinetic energy preservation across complex thermodynamic states, they were forced to abandon exact total energy conservation, opting instead to solve a non-conservative pressure evolution equation. Consequently, a numerical methodology that simultaneously guarantees full conservation—particularly of total energy—and exact pressure equilibrium for real and thermally perfect gases remains a missing link in the literature.

To address this critical gap, the present study introduces a novel numerical framework that, for the first time, achieves a fully conservative, kinetic-energy-preserving, and exactly pressure-equilibrium-preserving discretization of the compressible Euler equations for thermally perfect and real gases, with an arbitrary EoS. Our approach demonstrates that it is mathematically and practically possible to retain the exact discrete evolution of the total energy equation without triggering spurious interfacial oscillations, offering a robust and thermodynamically consistent foundation for simulating extreme real-gas phenomena. To overcome the intrinsic difficulties associated with simulations in transcritical conditions, an approximate formulation is also proposed, based on a slight modification of the set of exact fluxes. By retaining excellent PEP properties, with unrivaled performances among existing formulations, the set of approximate fluxes shows robust behavior also in extreme conditions.

The remainder of this paper is organized as follows. Section [2](#page-2-0) establishes the problem formulation, detailing the continuous compressible Euler equations and the underlying discretization framework. In Section [3,](#page-4-0) we present the core methodology, rigorously deriving the fully conservative numerical fluxes and proving their KEP and PEP properties at the discrete level. This generic formulation is subsequently particularized for specific thermodynamic models: Section [4](#page-7-0) addresses thermally perfect gases, while Section [5](#page-8-0) extends the approach to real gases, with a specific focus on the van der Waals and Peng–Robinson equations of state. Section [6](#page-10-0) presents a suite of numerical experiments—including a standard density wave test and a multidimensional mixing case—designed to validate both the PEP property and exact discrete conservation. Finally, Section [7](#page-16-0) summarizes the key findings and offers concluding remarks.

# 2 Problem formulation

# 2.1 Governing equations

The compressible Euler equations, which express the conservation of mass, momentum and total energy for compressible inviscid flows, can be written as:

$$\frac{\partial \rho}{\partial t} = -\frac{\partial \rho u_{\alpha}}{\partial x_{\alpha}} \,, \tag{1}$$

$$\frac{\partial \rho u_{\beta}}{\partial t} = -\frac{\partial \rho u_{\alpha} u_{\beta}}{\partial x_{\alpha}} - \frac{\partial p}{\partial x_{\beta}} , \qquad (2)$$

$$\frac{\partial \rho E}{\partial t} = -\frac{\partial \rho u_{\alpha} E}{\partial x_{\alpha}} - \frac{\partial \rho u_{\alpha}}{\partial x_{\alpha}},\tag{3}$$

where ρ is the density, u<sup>α</sup> is the Cartesian velocity component along xα, p is the pressure and E the total energy per unit mass, which is the sum of the internal (e) and kinetic (κ = uαuα/2) energies per unit mass: E = e + κ. We will assume the convention that Greek subscripts, such as α or β, refer to the components of Cartesian vectors, whereas Latin subscripts, such as i or j appearing in the subsequent sections of this paper, are used to denote the values of the discretized variable on a nodal point. Unless otherwise stated, the summation convention over repeated Greek indices is assumed. The system [\(1\)](#page-2-1)–[\(3\)](#page-2-2) is closed by an equation of state, relating p, ρ and the temperature T , and by specifying the dependence of internal energy on temperature and density, which is typically provided through suitable departure functions.

In what follows, we will also need to consider the induced balance equations for additional quantities related to the primary variables ρ, ρuβ, and ρE. These equations are easily derived by combining Eqs. [\(1\)](#page-2-1)–[\(3\)](#page-2-2) and by applying the chain and product rules of differentiation, with respect to both temporal and spatial variables. In particular, we will use the evolution equations for the kinetic and internal energies:

$$\frac{\partial \rho \kappa}{\partial t} = -\frac{\partial \rho u_{\alpha} \kappa}{\partial x_{\alpha}} - u_{\alpha} \frac{\partial p}{\partial x_{\alpha}},\tag{4}$$

$$\frac{\partial \rho e}{\partial t} = -\frac{\partial \rho u_{\alpha} e}{\partial x_{\alpha}} - p \frac{\partial u_{\alpha}}{\partial x_{\alpha}},\tag{5}$$

whose sum returns the total energy equation [\(3\)](#page-2-2). Moreover, to discuss the PEP property, which is the main subject of the present paper, we will use the evolution equations for the velocity components uβ, and the pressure p, which can be written as

$$\frac{\partial u_{\beta}}{\partial t} = -u_{\alpha} \frac{\partial u_{\beta}}{\partial x_{\alpha}} - \frac{1}{\rho} \frac{\partial p}{\partial x_{\beta}},\tag{6}$$

$$\frac{\partial p}{\partial t} = -\frac{\partial p u_{\alpha}}{\partial x_{\alpha}} - \left(\rho c^2 - p\right) \frac{\partial u_{\alpha}}{\partial x_{\alpha}},\tag{7}$$

where c = p (∂p/∂ρ)<sup>s</sup> is the speed of sound, s being the specific entropy. Eqs. [\(6\)](#page-2-3) and [\(7\)](#page-2-4) show that if at a given time instant a spatially constant state is assumed for both u<sup>β</sup> and p, all the spatial derivatives at the right-hand sides vanish. This also causes the temporal derivatives at the left-hand side to vanish, meaning that the constant distribution of pressure and velocity remains constant also in time. This is the Pressure Equilibrium property for the compressible Euler equations, which allows for the existence of density wave solutions traveling at constant velocity in a uniform pressure field.

#### 2.2 Discretization setting

Our focus is on the discretization of the spatial terms in Eqs. (1)–(3), for which we always use central (non-dissipative) schemes. To isolate spatial errors, we assume in the theoretical analysis that the manipulation of time derivatives can be performed as in the continuous case. This type of semidiscretized analysis leads to a system of Ordinary Differential Equations (ODE) whose temporal integration is performed by using standard solvers. In this paper, the theory will be developed for the one-dimensional version of Eq. (1)–(3), which is obtained by omitting the Greek subscripts. The extension to the general three-dimensional case is straightforward and involves simply including the analogous spatial terms along the additional Cartesian components. The analysis is developed by referring to a Finite Difference (FD) treatment of the convective terms in Eqs. (1)–(3) on a uniform mesh  $\{x_i\}$  of width  $h = x_{i+1} - x_i$ , although most of our results could also be applied in other frameworks, such as Finite Volume or Finite Element methods, and can be generalized to nonuniform or curvilinear meshes, following standard approaches [24, 25]. Finally, we focus on the discretization of the various spatial operators at internal points, neglecting the effects of boundary conditions, which could be studied with ad hoc methods.

Although we work in a FD framework, we will design the discretization of the convective spatial terms in Eqs. (1)–(3) by specifying the numerical fluxes, i.e. by assuming a *locally conservative* discretization in the form

$$\frac{\partial \rho u \phi}{\partial x} \bigg|_{i} \approx \frac{1}{h} \left( \mathcal{F}_{\rho \phi}^{i + \frac{1}{2}} - \mathcal{F}_{\rho \phi}^{i - \frac{1}{2}} \right) \tag{8}$$

where  $\phi$  is a generic transported quantity per unit mass (1, u or E in Eq. (1), (2) or (3), respectively). The numerical flux  $\mathcal{F}_{\rho\phi}^{i+\frac{1}{2}}$  is a consistent approximation of  $\rho u \phi$  at  $x_{i+1/2} = x_i + h/2$ , and is a function of the sets of nodal values  $\rho_i, u_i$  and  $\phi_i$ . Note that, even if the right-hand sides of Eq. (1)–(3) can be expressed as the divergence of a single term including convective and pressure mechanisms, Eq. (8) refers to the specification of the numerical fluxes for the convective contribution only (the first terms at the r.h.s. of Eqs. (1)–(3)). The terms involving the pressure in Eq. (2) and (3) are discretized by using standard central derivative formulas, as specified in Section 2.3. Eq. (8) reproduces the divergence structure of the convective terms in Eqs. (1)–(3) and implies also global conservation of the invariant  $\rho\phi$  by virtue of the telescoping property.

To simplify the notation, we will make use of the difference and average operators

$$\delta^- \varphi_i = \varphi_i - \varphi_{i-1}, \qquad \delta^+ \varphi_i = \varphi_{i+1} - \varphi_i, \qquad \overline{\varphi}_i = (\varphi_{i+1} + \varphi_i)/2$$

which allows to write the central difference as  $\varphi_{i+1} - \varphi_{i-1} = 2\delta^{-}\overline{\varphi}_{i}$  and to express Eq. (8) as

$$\left. \frac{\partial \rho u \phi}{\partial x} \right|_{i} \approx \frac{1}{h} \delta^{-} \mathcal{F}_{\rho \phi}^{i + \frac{1}{2}}.$$

In most cases, we will omit the apex when referring to the flux evaluated at  $x_{i+1/2}$ :  $\mathcal{F}_{\rho\phi} = \mathcal{F}_{\rho\phi}^{i+\frac{1}{2}}$ , and, when no ambiguity can arise, we also omit the suffix in the expressions of the difference or of the average of a generic quantity:  $\overline{\phi} = \overline{\phi}_i$ ,  $\delta^-\phi = \delta^-\phi_i$ .

Within the described setting, the spatial discretization of the r.h.s. of Eqs. (1)–(3) is completely determined by the specification of the convective fluxes  $\mathcal{F}_{\rho}$ ,  $\mathcal{F}_{\rho u}$ ,  $\mathcal{F}_{\rho E}$  and by the discretization details of the pressure terms  $\partial p/\partial x$  and  $\partial up/\partial x$  in Eqs. (2) and (3), respectively. The derivation exposed in the next sections considers second-order approximations, which implies that the numerical flux  $\mathcal{F}_{\rho\phi}$  is in general a two-point interpolation of  $(\rho u\phi)_{i+1/2}$ , and is a function of  $\rho$ , u and  $\phi$  at nodes  $x_i$  and  $x_{i+1}$  only. The extension to higher orders is obtained by using standard arguments and is detailed in several references [26, 27, 28, 21, 29].

### 2.3 Kinetic Energy Preserving formulation

A first constraint on the possible choice for the numerical fluxes is given by the requirement that the formulation satisfies the Kinetic Energy Preserving (KEP) property [15, 16, 30, 31], which amounts to the requirement that the discrete induced evolution equation for the kinetic energy has a convective term that is locally and globally conservative, as for the continuous equation (cf. Eq. (4)). It is well known ([16, 32, 33]) that, for second order fluxes, the KEP property is satisfied when the convective fluxes for mass and momentum are linked by the simple relation

$$\mathcal{F}_{\rho u} = \mathcal{F}_{\rho} \, \overline{u}$$
.

In this case, kinetic energy evolves according to a discrete equation analogous to Eq. (4), in which the convective term can be cast as a difference of numerical fluxes, with flux [33, 34]

$$\mathcal{F}_{\rho\kappa} = \mathcal{F}_{\rho} \frac{u_i u_{i+1}}{2}.\tag{9}$$

The requirement that the discrete total energy evolves coherently with the induced evolution of kinetic energy suggests designing the convective numerical flux for total energy  $\mathcal{F}_{\rho E}$  as the sum of the kinetic-energy flux given by Eq. (9) and of a convective flux for internal energy  $\mathcal{F}_{\rho e}$ . The conservative pressure term is correspondingly built as the sum of the induced discretized pressure term in Eq. (4) (which belongs to the discrete pressure term in the momentum equation (2)) and of a consistent discretization of the pressure term in Eq. (5). Assuming standard second-order central schemes for these terms, one has

$$u_i \frac{\partial p}{\partial x}\Big|_i \approx u_i \frac{1}{h} \delta^{-} \overline{p}$$
  $p_i \frac{\partial u}{\partial x}\Big|_i \approx p_i \frac{1}{h} \delta^{-} \overline{u}$  (10)

which gives the correct discretization of the pressure term in the total energy equation as consistent with the advective form of the derivative of the product:

$$\frac{\partial up}{\partial x}\bigg|_{i} = u_{i} \frac{\partial p}{\partial x}\bigg|_{i} + p_{i} \frac{\partial u}{\partial x}\bigg|_{i} \approx \frac{1}{h} \left(u_{i}\delta^{-}\overline{p} + p_{i}\delta^{-}\overline{u}\right) = \frac{1}{h}\delta^{-}\overline{u}\overline{p}$$

where  $\overline{\overline{up}} = (u_i p_{i+1} + p_i u_{i+1})/2$  is the product mean [31].

In conclusion, a locally conservative and KEP discretization is characterized by the set of total fluxes (i.e. including both convective and pressure contributions) given by

$$\mathcal{F}_{\rho}^{\text{tot}} = \mathcal{F}_{\rho}, \qquad \qquad \mathcal{F}_{\rho u}^{\text{tot}} = \mathcal{F}_{\rho} \, \overline{u} + \overline{p}, \qquad \qquad \mathcal{F}_{\rho E}^{\text{tot}} = \mathcal{F}_{\rho e} + \mathcal{F}_{\rho} \frac{u_{i} u_{i+1}}{2} + \overline{\overline{u} p}, \tag{11}$$

where the mass and internal energy fluxes  $\mathcal{F}_{\rho}$  and  $\mathcal{F}_{\rho e}$  are still unspecified, except for the obvious requirement that they are consistent approximations of mass and internal energy fluxes. The set of fluxes reported in Eq. (11) defines a formulation which is locally conservative of linear invariants  $\rho, \rho u$  and  $\rho E$  and preserves the (local and global) conservation of kinetic energy by convection. It embodies many KEP formulations available in the literature that have gained popularity in recent years. As examples, the simple choice  $\mathcal{F}_{\rho} = \overline{\rho} \, \overline{u}$  and  $\mathcal{F}_{\rho e} = \overline{\rho} \, \overline{u} \, \overline{e}$  gives the KEEP scheme [35] which, even if it does not have exact additional structural properties, has been widely used because of its good performances for ideal gases. The choice  $\mathcal{F}_{\rho} = \overline{\rho} \, \overline{u}$  and  $\mathcal{F}_{\rho e} = \overline{\rho} \, \overline{u} \, \overline{e}^H$  (where  $\overline{e}^H = 2e_i e_{i+1}/(e_i + e_{i+1})$  is the harmonic mean) gives the zeroth-order AEC scheme, developed in [20], whereas  $\mathcal{F}_{\rho} = \overline{\rho} \, \overline{u}$  and  $\mathcal{F}_{\rho e} = \overline{\rho} \, \overline{u}$  give the KEEP<sub>PE</sub> scheme [18], these last two formulations being PEP for ideal gases. Recently, the present authors also determined mass and internal energy fluxes such that the formulation in Eq. (11) is Entropy Conservative (EC), i.e. it is able to discretely preserve the correct balance of entropy, for the case of real gases with an arbitrary equation of state [29] and for thermally perfect gases [36]. In the next section, we derive an expression for  $\mathcal{F}_{\rho}$  and  $\mathcal{F}_{\rho e}$  which gives a formulation able to satisfy the PEP condition for thermally perfect and real gases with an arbitrary EoS.

## 3 PEP formulation

In this section, we analyze the general problem of enforcing the PEP condition for the system of compressible Euler equations, without making any assumptions about the equation of state. In Sections 4 and 5 we explicitly work out the particular cases of a thermally perfect gas and of various thermodynamic models for real gases.

#### 3.1 Theoretical derivation of exact PEP fluxes

In order to discretely enforce the pressure equilibrium property, it is necessary to consider the discrete evolution equations for velocity components and pressure, which are the induced discrete counterparts of Eqs. (6) and (7). Among these, the pressure evolution equation is the more challenging one, due to its explicit dependence on the EoS. We therefore begin with the discrete velocity equation, which is independent of the EoS. General conditions for the discrete enforcement of the velocity equilibrium have previously been derived for the case of calorically perfect gases [21, 19], where it was shown that the KEP condition is sufficient to enforce it. This result still applies but, for completeness, we briefly revisit the derivation here.

First of all, we explicitly write the semidiscrete evolution equations for mass and momentum as obtained by discretizing Eqs. (1) and (2) according to the prescriptions illustrated in the previous section:

$$\frac{\mathrm{d}\rho_i}{\mathrm{d}t} = -\frac{1}{b}\delta^- \mathcal{F}_\rho \,\,\,\,(12)$$

$$\frac{\mathrm{d}\rho_i u_i}{\mathrm{d}t} = -\frac{1}{h} \delta^- \mathcal{F}_\rho \overline{u} - \frac{1}{h} \delta^- \overline{p} \ . \tag{13}$$

The discrete evolution equation for the velocity can be now obtained by using the expression of the time derivative of the velocity u in terms of the time derivatives of  $\rho$  and  $\rho u$ 

$$\frac{\partial u}{\partial t} = \frac{1}{\rho} \frac{\partial \rho u}{\partial t} - \frac{u}{\rho} \frac{\partial \rho}{\partial t}.$$
 (14)

From this relation, substituting Eqs. (12) and (13), one obtains

$$\frac{\mathrm{d}u_i}{\mathrm{d}t} = -\frac{1}{\rho_i h} \left( \delta^- \mathcal{F}_{\rho} \overline{u} + \delta^- \overline{p} - u_i \, \delta^- \mathcal{F}_{\rho} \right). \tag{15}$$

In the case of spatially constant pressure  $p_i = P$  and velocity  $u_i = U$ , the pressure contribution at the right-hand side vanishes, and one is left with

$$\frac{\mathrm{d}u_i}{\mathrm{d}t} = -\frac{1}{\rho_i h} \left( U \delta^- \hat{\mathcal{F}}_\rho - U \delta^- \hat{\mathcal{F}}_\rho \right) = 0, \tag{16}$$

where we denoted with  $\mathcal{F}_{\rho}$  the form assumed by  $\mathcal{F}_{\rho}$  for uniform velocity and pressure fields. Eq. (16) shows that for the KEP discretization adopted, a uniform spatial distribution of pressure and velocity always induces a zero time derivative for  $u_i$ , which implies that the velocity field remains uniform as time evolves.

Moving on to the discrete evolution of pressure, we start by considering the internal energy per unit volume  $\rho e$  and use an equation of state in the form  $\rho e = \rho e(\rho, p)$ . By taking the time derivative of this relation, we get

$$\frac{\partial \rho e}{\partial t} = \left(\frac{\partial \rho e}{\partial \rho}\right)_p \frac{\partial \rho}{\partial t} + \left(\frac{\partial \rho e}{\partial p}\right)_\rho \frac{\partial p}{\partial t}.$$

Substituting the temporal derivatives with the right-hand sides of the mass and internal energy equations (1) and (5), we obtain

$$\frac{\partial \rho ue}{\partial x} + p \frac{\partial u}{\partial x} = \alpha(\rho, p) \frac{\partial \rho u}{\partial x} + \beta(\rho, p) \frac{\partial p}{\partial t}.$$

where we defined

$$\alpha(\rho,p) = \left(\frac{\partial \rho e}{\partial \rho}\right)_p \qquad \text{and} \qquad \beta(\rho,p) = \left(\frac{\partial \rho e}{\partial p}\right)_\rho.$$

Now we assume to discretize the convective terms in the mass and internal energy equations conservatively, i.e. as the difference of numerical fluxes  $\delta^- \mathcal{F}/h$ , and to use the simple central discretization for the pressure term in the internal energy equation given in Eq. (10). This gives

$$\frac{1}{h}\delta^{-}\mathcal{F}_{\rho e} + p_{i}\,\frac{1}{h}\delta^{-}\overline{u} = \alpha(\rho_{i}, p_{i})\frac{1}{h}\delta^{-}\mathcal{F}_{\rho} + \beta(\rho_{i}, p_{i})\frac{\mathrm{d}p_{i}}{\mathrm{d}t}.$$

Assuming again uniform spatial distributions of pressure and velocity, one is left with

$$\delta^{-}\hat{\mathcal{F}}_{\rho e} = \alpha(\rho_{i}, P) \,\delta^{-}\hat{\mathcal{F}}_{\rho} + h \,\beta(\rho_{i}, P) \frac{\mathrm{d}p_{i}}{\mathrm{d}t},\tag{17}$$

where, as usual,  $\hat{\mathcal{F}}_{\rho e}$  and  $\hat{\mathcal{F}}_{\rho}$  denote the form assumed by the numerical fluxes for uniform velocity and pressure fields. Eq. (17) reveals that to discretely maintain the uniform pressure distribution (i.e. to enforce the condition  $\mathrm{d}p_i/\mathrm{d}t=0$ ), the mass and internal energy fluxes have to satisfy the constraint

$$\delta^{-}\hat{\mathcal{F}}_{\rho e} = \hat{\alpha}_{i}\,\delta^{-}\hat{\mathcal{F}}_{\rho},\tag{18}$$

where  $\hat{\alpha}_i = \hat{\alpha}(\rho_i) = \alpha(\rho_i, P)$ .

A condition analogous to that in Eq. (18) has already been analyzed in [13] for the case of a multicomponent mixture of real gases and has been the starting point to obtain approximate PEP formulations in [14]. An exact enforcement of Eq. (18) seems problematic at first sight, since it requires the specification of the numerical fluxes  $\mathcal{F}_{\rho}$  and  $\mathcal{F}_{\rho e}$  such that the product between the exact difference  $\delta^{-}\hat{\mathcal{F}}_{\rho}$  and an arbitrary function  $\hat{\alpha}(\rho)$  can be expressed as the difference of numerical fluxes  $\delta^{-}\hat{\mathcal{F}}_{\rho e}$ . In Eq. (18), the function  $\hat{\alpha}(\rho)$  is fixed by the thermodynamic model, whereas  $\hat{\mathcal{F}}_{\rho}$  and  $\hat{\mathcal{F}}_{\rho e}$  can be chosen arbitrarily, provided that they are consistent approximations of the mass and internal energy fluxes.

To solve the problem expressed by Eq. (18) we proceed inspired by the steps of the theory of Tadmor [37, 38] and observe that  $\hat{\alpha}_i \delta^- \hat{\mathcal{F}}_\rho$  is an exact difference if (and only if)  $\hat{\mathcal{F}}_\rho \delta^+ \hat{\alpha}_i$  is, since the sum  $a_i \delta^- b_i + b_i \delta^+ a_i$  (which is a consistent discretization of the advective form of the derivative of the product ab) admits the decomposition as a difference of fluxes  $a_i \delta^- b_i + b_i \delta^+ a_i = \delta^- (a_{i+1} b_i)$ . This allows us to rewrite Eq. (18) in the equivalent form

$$\delta^{-}\hat{\mathcal{F}}_{\rho e} = -\hat{\mathcal{F}}_{\rho}\delta^{+}\hat{\alpha}_{i} + \delta^{-}\left(\hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho}\right),\,$$

and we are now faced with the problem of finding a numerical flux  $\mathcal{F}_{\rho}$  such that  $\hat{\mathcal{F}}_{\rho}\delta^{+}\hat{\alpha}_{i}$  is an exact difference. We proceed now by observing that the flux  $\mathcal{F}_{\rho}$  should be a consistent approximation of the analytical flux  $f_{\rho}(\rho,u)=\rho u$  at  $x_{i+1/2}$ . We assume that the function  $\hat{\alpha}(\rho)$  is (at least locally) invertible, in such a way that the inverse function  $\rho(\hat{\alpha})$  can be considered, and the relation  $\rho-\hat{\alpha}$  is a local one-to-one mapping. This confers to the variable  $\hat{\alpha}$  the role of an entropy variable, in the terminology of the theory of Tadmor. In this case we can express the function  $\hat{f}_{\rho}(\rho)=f_{\rho}(\rho,U)$  through the new variable  $\hat{\alpha}$  by defining the function  $\hat{g}_{\rho}(\hat{\alpha})=\hat{f}_{\rho}(\rho(\hat{\alpha}))$ , which is approximated by the numerical flux  $\hat{\mathcal{G}}_{\rho}(\hat{\alpha}_{i},\hat{\alpha}_{i+1})=\hat{\mathcal{F}}_{\rho}(\rho(\hat{\alpha}_{i}),\rho(\hat{\alpha}_{i+1}))$ .

To express now  $\hat{\mathcal{G}}_{\rho}\delta^{+}\hat{\alpha}$  as an exact difference, we make use of the primitive function  $\psi(\hat{\alpha})$  defined by

$$\psi(\hat{\alpha}) = \int \hat{g}_{\rho}(\hat{\alpha}) \, \mathrm{d}\hat{\alpha}.$$

Using the Integral Mean Value Theorem, we can write

$$\delta^{+}\psi_{i} = \int_{\hat{\alpha}_{i}}^{\hat{\alpha}_{i+1}} \hat{g}_{\rho}(\hat{\alpha}) \,\mathrm{d}\hat{\alpha} = \hat{g}_{\rho}(\hat{\alpha}^{*}) \,\delta^{+}\hat{\alpha} \tag{19}$$

where  $\hat{\alpha}^*$  is a unknown value between  $\hat{\alpha}_i$  and  $\hat{\alpha}_{i+1}$  and the value  $\hat{g}_{\rho}(\hat{\alpha}^*)$  is a second-order approximation of  $\hat{g}_{\rho}(\hat{\alpha}_{i+\frac{1}{2}})$ . Eq. (19) suggests the adoption of the value  $\hat{\mathcal{G}}_{\rho} = \hat{g}_{\rho}(\hat{\alpha}^*)$ , i.e.

$$\hat{\mathcal{G}}_{\rho}(\hat{\alpha}_i, \hat{\alpha}_{i+1}) = \frac{\delta^{+}\psi_i}{\delta^{+}\hat{\alpha}_i},$$

which, expressed in the variables  $\rho_i$  gives:

$$\hat{\mathcal{F}}_{\rho}(\rho_i, \rho_{i+1}) = \hat{\mathcal{G}}_{\rho}(\hat{\alpha}(\rho_i), \hat{\alpha}(\rho_{i+1})) = \frac{\delta^+ \psi_i}{\delta^+ \hat{\alpha}_i}$$
(20)

where  $\psi(\rho) = \psi(\hat{\alpha}(\rho))$ . This gives:

$$\hat{\alpha}_{i}\delta^{-}\hat{\mathcal{F}}_{\rho} = -\hat{\mathcal{F}}_{\rho}\delta^{+}\hat{\alpha}_{i} + \delta^{-}\left(\hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho}\right) = -\delta^{+}\psi_{i} + \delta^{-}\left(\hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho}\right) = \delta^{-}\left(\hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho} - \psi_{i+1}\right),\tag{21}$$

where we used  $\delta^+\psi_i = \delta^-\psi_{i+1}$ . Eq. (21) reduces to Eq. (18) with the choice  $\hat{\mathcal{F}}_{\rho e} = \hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho} - \psi_{i+1}$ . The form of  $\hat{\mathcal{F}}_{\rho e}$  can be made more symmetric by observing that  $\hat{\alpha}_{i+1} = \overline{\hat{\alpha}}_i + \delta^+\hat{\alpha}_i/2$  which implies

$$\hat{\alpha}_{i+1}\hat{\mathcal{F}}_{\rho} = \overline{\hat{\alpha}}_{i}\hat{\mathcal{F}}_{\rho} + \frac{\hat{\mathcal{F}}_{\rho}\delta^{+}\hat{\alpha}_{i}}{2} = \overline{\hat{\alpha}}_{i}\hat{\mathcal{F}}_{\rho} + \frac{\delta^{+}\psi_{i}}{2}$$

eventually giving

$$\hat{\mathcal{F}}_{\rho e} = \overline{\hat{\alpha}}_i \hat{\mathcal{F}}_{\rho} - \overline{\psi}_i. \tag{22}$$

Equations (20) and (22) satisfy Eq. (18), and constitute the basis to build the general fluxes  $\mathcal{F}_{\rho}$  and  $\mathcal{F}_{\rho e}$  enforcing the PEP property.

To show how the function  $\psi$  and the fluxes in Eqs. (20) and (22) can be practically calculated, let us define the function

$$\lambda(\rho, p) = \left(\frac{\partial e}{\partial \rho}\right)_{p} \tag{23}$$

which is related to  $\alpha$  by the equation  $\alpha = e + \rho \lambda$ . By using  $f_{\rho}(\rho, u) = \rho u$  the function  $\psi(\rho)$  can be calculated as

$$\psi(\rho) = \psi(\alpha(\rho)) = \int \hat{g}_{\rho}(\hat{\alpha}(\rho)) \,d\hat{\alpha}(\rho) = \int \hat{f}_{\rho}(\rho)\hat{\alpha}' \,d\rho = U \int \rho \left(2\hat{\lambda} + \rho\hat{\lambda}'\right) d\rho = U\rho^{2}\hat{\lambda}$$
(24)

where primes denote differentiation and, as usual,  $\hat{\lambda}(\rho) = \lambda(\rho, U)$ . Eq. (24) allows us to write the fluxes satisfying Eq. (18) in the explicit form

$$\hat{\mathcal{F}}_{\rho} = \frac{\delta^{+} \rho_{i}^{2} \hat{\lambda}_{i}}{\delta^{+} \hat{\alpha}_{i}} U \qquad \qquad \hat{\mathcal{F}}_{\rho e} = \overline{\hat{\alpha}} \ \hat{\mathcal{F}}_{\rho} - U \overline{\rho^{2} \hat{\lambda}}$$
 (25)

The final form of  $\mathcal{F}_{\rho}$  and  $\mathcal{F}_{\rho e}$  can now be obtained in several ways, all reducing to Eq (25) for uniform velocity. In what follows we use the simple extension of Eq. (25) obtained by adopting the arithmetic mean for u, leading to:

$$\mathcal{F}_{\rho} = \overline{\rho}^{\lambda} \overline{u} \qquad \qquad \mathcal{F}_{\rho e} = \overline{\alpha} \, \mathcal{F}_{\rho} - \overline{u} \, \overline{\rho^{2} \lambda} \,. \tag{26}$$

where we defined the average  $\overline{\rho}^{\,\lambda}=\delta^+(\rho_i^2\lambda_i)/\delta^+\alpha_i.$ 

#### 3.2 Treatment of the singularity and approximate PEP formulation

Equations (26) and (11) furnish the final formulation satisfying the KEP and PEP properties for an arbitrary EoS, whose details are embodied in the functions  $\alpha(\rho,p)$  and  $\lambda(\rho,p)$ . As it is typical in these cases, the nonlinear average  $\overline{\rho}^{\lambda}$  in the mass flux is potentially singular for uniform distributions of  $\alpha$ . A similar situation occurs, for example, in the EC and PEP (for ideal gases) flux by Ranocha [39], which employs the logarithmic mean  $(\overline{\phi}^{\log} = \delta^+ \phi/\delta^+ \log \phi)$ , and in the EC flux for real gases recently developed in [29], which also uses a potentially singular flux for uniform distributions of temperature. This phenomenon, which is fundamentally linked to the definition of the average through the use of the integral mean value theorem [40], can cause severe limitations in the applications, especially when large regions of uniform distributions of temperature or density are expected to occur. To avoid the singularity, suitable fixes can be devised, typically implemented by locally reverting to non-singular numerical fluxes when the denominators of the singular averages ( $\delta^+\alpha_i$  in our case) fall under a specific tolerance. These non-singular schemes can be either a suitable Taylor expansion of the original mean, when possible, as in the case of the logarithmic mean used in entropy conservative methods for ideal gases ([41, 39, 20, 22]) or standard non-singular schemes chosen among those with good performances, to minimize errors, as in [29]. Recently, a more advanced technique based on the theory of discrete gradient operators has been also used [42].

In the present case, since a full series expansion of the singular mean  $\overline{\rho}^{\lambda}$  appearing in the mass flux  $\mathcal{F}_{\rho}$  seems cumbersome for an arbitrary EoS, we choose to use the simple modification of the flux  $\mathcal{F}_{\rho}$  in Eq. (26) obtained by adopting the arithmetic mean  $\overline{\rho}$  in place of  $\overline{\rho}^{\lambda}$ , i.e. by using the fluxes

$$\mathcal{F}_{\rho} = \overline{\rho} \, \overline{u} \qquad \qquad \mathcal{F}_{\rho e} = \overline{\alpha} \, \mathcal{F}_{\rho} - \overline{u} \, \overline{\rho^2 \lambda} \,. \tag{27}$$

The set of fluxes given by Eqs. (11) and (27) actually shows excellent performances on highly challenging tests, as reported in Section 6. These results tempted us to use it not only as a fix for the exact PEP formulation given by Eq. (26), but also as an approximate PEP formulation which can be used during the whole simulation, irrespective of the local value of  $\delta^+\alpha_i$ . Similar very good performances have also been observed by using other classical algebraic means for the density in the mass flux (e.g. the harmonic or geometric means), suggesting that the form of the interpolation for the density in the mass flux has only marginal effects on the enforcement of the PEP property. For the sake of simplicity, we adopt here the arithmetic mean, leaving a thorough analysis of this subject as a possible future work. In conclusion, although only approximately PEP, the excellent performances of the formulation (11)–(27) induce us to offer it as a sufficiently simple formulation which could be used in place of Eqs. (11)–(26). We will refer to the Exact PEP formulation (EPEP-RG) for the scheme given by Eqs. (11)–(26), and to the Approximate PEP formulation (APEP-RG) for that given by Eqs. (11)–(27).

As a final comment, we observe that in the case of a calorically perfect gas, for which  $\rho e = p/(\gamma - 1)$ , the function  $\alpha$  is identically zero, and the whole derivation exposed in Section 3.1, that led to Eq. (26), breaks down. However, even if the mass flux in Eq. (26) remains undefined, the internal energy flux can be safely determined, and turns out to reduce to  $\mathcal{F}_{\rho e} = \overline{\rho e} \, \overline{u}$ , which is the internal energy flux of the formulation KEEP<sub>PE</sub> by Shima et al. [18]. Hence, the APEP-RG formulation defined by the fluxes in Eq. (27) nicely reduces to the KEEP<sub>PE</sub> formulation in the case of a calorically perfect gas.

### 4 Thermally perfect gases

In this section, we work out the particular case of a thermally perfect gas model, for which the usual perfect-gas EoS is assumed:  $p = \rho RT$ , where R is the gas constant and T is the absolute temperature. The internal energy depends on temperature through temperature-variable isochoric specific heat capacity  $c_v(T)$ , which implies

$$e = \int_{T_{\text{ref}}}^{T} c_v(T') \, \mathrm{d}T' + e_{\text{ref}}$$
 (28)

and the "ref" subscript indicates some reference condition. We use a polynomial-based approach for the modeling of  $c_v(T)$ , [43], by which the isochoric specific heat is expressed using temperature-based polynomial fittings:

$$c_v(T) = \sum_{k=0}^{N} c_k T^k. \tag{29}$$

This functional dependence is widely used when thermal equilibrium is assumed [43], and is also easily tractable from an analytical point of view. We will detail the derivation for the general formulation with arbitrary N, although only

5, 7 or 9 coefficients are typically used to experimentally fit the gas behaviour [44, 45]. By substituting Eq. (29) into Eq. (28) and using the perfect-gas EoS one has

$$\alpha(\rho, p) = -\sum_{k=1}^{N} A_k T^{k+1} + \varepsilon_{\text{ref}}, \qquad \lambda(\rho, p) = -\sum_{k=1}^{N} c_k \frac{T^{k+1}}{\rho}$$
(30)

with  $A_k = kc_k/(k+1)$  and  $T(\rho, p) = p/\rho R$ . The final form of the PEP fluxes corresponding with the fluxes in Eq. (26) for a thermally perfect gas is

$$\mathcal{F}_{\rho} = \frac{\delta^{+} \sum_{k=1}^{N} c_{k} \rho T^{k+1}}{\delta^{+} \sum_{k=1}^{N} A_{k} T^{k+1}} \overline{u} \qquad \qquad \mathcal{F}_{\rho e} = - \overline{\left(\sum_{k=1}^{N} A_{k} T^{k+1}\right)} \, \mathcal{F}_{\rho} + \overline{u} \, \overline{\left(\sum_{k=1}^{N} c_{k} \rho T^{k+1}\right)}. \tag{31}$$

As for the general form of the fluxes in Eq. (26), the mass flux for thermally perfect gases in Eq. (31) is potentially singular. However, the particular polynomial form assumed for the  $c_v(T)$  allows one to derive a formulation that is singularity-free in all conditions, without the need for a fix. To obtain this formulation, we need to derive the general form of the mass flux by starting from the PEP condition in Eq. (25), which, using Eq. (30), can be written, for thermally perfect gases, as

$$\hat{\mathcal{F}}_{\rho} = \frac{\sum_{k=1}^{N} c_k \delta^+ T^k}{\sum_{k=1}^{N} A_k \delta^+ T^{k+1}} \frac{PU}{R}$$

where we used the ideal gas equation of state particularized to the case of uniform pressure:  $\rho = P/RT$ . Using now the general identity

$$\delta^{+}\phi^{k} = \delta^{+}\phi \sum_{j=0}^{k-1} \phi_{i}^{k-1-j} \phi_{i+1}^{j},$$

one finally gets

$$\hat{\mathcal{F}}_{\rho} = \frac{\sum_{k=1}^{N} c_k \sum_{j=0}^{k-1} T_i^{k-1-j} T_{i+1}^j}{\sum_{k=1}^{N} A_k \sum_{j=0}^{k} T_i^{k-j} T_{i+1}^j} \frac{PU}{R},$$

for which the denominator does not vanish. The final form of the flux can now be obtained by assuming an arithmetic interpolation for pressure and velocity:

$$\mathcal{F}_{\rho} = \frac{\sum_{k=1}^{N} c_k S_i^k}{\sum_{k=1}^{N} A_k S_i^{k+1}} \frac{\overline{p} \ \overline{u}}{R}$$
(32)

where  $S_i^k = \sum_{j=0}^{k-1} T_i^{k-1-j} T_{i+1}^j$ . The flux can be calculated more efficiently by observing the identity  $S_i^{k+1} = T_i S_i^k + T_{i+1}^k$ . The mass flux in Eq. (32), together with the internal energy flux in Eq. (31), is a more efficient PEP formulation for thermally perfect gases, and is the one adopted in the numerical tests in Section 6. Note that in practical applications, the polynomial fitting for the  $c_v$  in Eq. (29) can also include terms with negative exponents. The treatment can be easily generalized to this case as in [36], and the fluxes in Eqs. (31) and (32) are consequently adapted.

## 5 Real gases

For real gases, we adopt the usual representation of the internal energy by means of departure functions, whose expressions are given, for example, in [46] assuming the form

$$e(\rho, T) = e^{\text{TP}}(T) + D^{e}(\rho, T) = \int_{T_{\text{ref}}}^{T} c_{v}(T') dT' + e_{\text{ref}} + D^{e}(\rho, T),$$
(33)

where  $e^{\mathrm{TP}}$  is the thermally perfect contribution given in Eq. (28) and  $D^e$  is the suitable departure function for internal energy [46]. In general, we are denoting with  $D^{\phi}$  the departure function for the thermodynamic quantity  $\phi$ , which is defined by  $D^{\phi} = \phi - \phi^{\mathrm{TP}}$  and  $\phi^{\mathrm{TP}}$  is the thermally perfect contribution.

Standard manipulations give

$$\lambda(\rho, p) = \left(\frac{\partial e}{\partial \rho}\right)_p = \left(\frac{\partial e^{\text{TP}}}{\partial \rho}\right)_p + \left(\frac{\partial D^e}{\partial \rho}\right)_p = c_v(T) \left(\frac{\partial T}{\partial \rho}\right)_p + \left(\frac{\partial D^e}{\partial \rho}\right)_T + \left(\frac{\partial D^e}{\partial T}\right)_\rho \left(\frac{\partial T}{\partial \rho}\right)_p, \tag{34}$$

where  $T = T(\rho, p)$  is the explicit form of the equation of state. Gathering common terms and noting that, by definition,  $(\partial D^e/\partial T)_{\rho} = D^{c_v}(\rho, T)$ , yields

$$\lambda(\rho, p) = c_v^{\text{RG}}(\rho, T) \left(\frac{\partial T}{\partial \rho}\right)_p + \left(\frac{\partial D^e}{\partial \rho}\right)_T, \qquad \alpha(\rho, p) = e + \rho \left(c_v^{\text{RG}}(\rho, T) \left(\frac{\partial T}{\partial \rho}\right)_p + \left(\frac{\partial D^e}{\partial \rho}\right)_T\right). \tag{35}$$

Finally, for the evaluation of the speed of sound c, required by the use of the pressure evolution equation in the main Euler system, we will always use the standard expression (particularized for each equation of state)

$$c^{2} = \frac{1}{\beta_{s}} = \frac{1}{\beta_{T}} + \frac{T\left[\left(\partial p/\partial T\right)_{\rho}\right]^{2}}{\rho c_{s}^{RG}}, \quad \text{with} \quad \beta_{T} = \frac{1}{\rho\left(\partial p/\partial \rho\right)_{T}}$$
(36)

 $\beta_s$ ,  $\beta_T$  being the isentropic and isothermal compressibility, respectively.

In the following sections, specializations for the van der Waals and Peng–Robinson models of Eq. (35) will be derived in non-dimensional form.

### 5.1 Van der Waals model

The van der Waals model is taken into account due to its simplicity as a first simple real-gas correction to the ideal gas equation of state. The equation of state for pressure reads

$$p = \frac{\rho T}{1 - \rho b} - a\rho^2 \tag{37}$$

with  $a^*=(27/64)(R^*T_c^*)^2/p_c^*$  and  $b^*=(1/8)R^*T_c^*/p_c^*$ , with the superscript \* indicating dimensional values and the suffix 'c' referring to critical conditions. Internal energy is given by

$$e = e^{\text{TP}} + D^e = \int_{T_{\text{ref}}}^{T} c_v(T') dT' - a\rho$$
 (38)

with  $D^e = -a\rho$ . With such definitions, Eq. (35) becomes

$$\lambda(\rho, p) = \left(c_v(T)\left(a - \frac{p}{\rho^2} - 2a\rho b\right) - a\right), \qquad \alpha(\rho, p) = e + \rho\left(c_v(T)\left(a - \frac{p}{\rho^2} - 2a\rho b\right) - a\right), \tag{39}$$
where  $c_v(T) = c_v^{\text{RG}}(T)$ , since  $D^{c_v} = \int_0^\rho \rho^{-2} T(\partial_{TT}^2 p)|_\rho \, \mathrm{d}\rho = 0$ .

## 5.2 Peng-Robinson model

The Peng-Robinson model is considered because of its widespread application in high-pressure/low-temperature flows, as it overcomes the intrinsic instability of the van der Waals model near the critical region. The equation of state for pressure is

$$p = \frac{\rho T}{(1 - \rho b)} - \frac{\rho^2 a A(T)}{1 + 2\rho b - (\rho b)^2}$$
(40)

with

$$A(T) = \left(1 + k \left(1 - \sqrt{\frac{TT_{\text{ref}}^*}{T_c^*}}\right)\right)^2, \qquad k = 0.37464 + 1.54226\omega - 0.26992\omega^2 \tag{41}$$

and  $a^*=0.45724(R^*T_c^*)^2/p_c^*$ ,  $b^*=0.0778(R^*T_c^*/p_c^*)$ . A(T) is the Soave function for accounting temperature dependence on potential energy, while  $\omega=0.2249$  is the acentric factor for  ${\rm CO}_2$ . Internal energy has the expression

$$e = e^{\text{TP}} + D^e = \int_{T_{\text{ref}}}^T c_v(T') \, dT' - \frac{a(TA'(T) - A(T))}{2\sqrt{2}b} \log\left(\frac{1 + (1 + \sqrt{2})\rho b}{1 + (1 - \sqrt{2})\rho b}\right). \tag{42}$$

For this specific model,  $D^{c_v} \neq 0$ , and is

$$D^{c_v} = \int_0^\rho \rho^{-2} T\left(\frac{\partial^2 p}{\partial T^2}\right)_a d\rho = \frac{TaA''(T)}{2\sqrt{2}b} \log\left(\frac{1 + (1 + \sqrt{2})\rho b}{1 + (1 - \sqrt{2})\rho b}\right). \tag{43}$$

The expression for  $\lambda(\rho, p)$  is calculated by using Eq. (35) with

$$\left(\frac{\partial T}{\partial \rho}\right)_p = -\frac{(\partial p/\partial \rho)_T}{(\partial p/\partial T)_\rho} \quad \text{and} \quad \left(\frac{\partial D^e}{\partial \rho}\right)_T = \frac{a(TA'(T) - A(T))}{1 + 2\rho b - (\rho b)^2}$$

| Scheme             | Ref.                  | ${\cal F}_{\rho}$                       | ${\cal F}_{\rho e}$                                                                          | PEP IG | PEP RG | Fully Cons. |
|--------------------|-----------------------|-----------------------------------------|----------------------------------------------------------------------------------------------|--------|--------|-------------|
| EPEP-RG            | new                   | $\overline{\rho}^{\lambda}\overline{u}$ | $\overline{\alpha}\mathcal{F}_{\rho}-\overline{u}\overline{\rho^2\lambda}$                   | n.a.   | ✓      | ✓           |
| APEP-RG            | new                   | $\overline{\rho} \overline{u}$          | $\overline{\alpha}\mathcal{F}_{\rho} - \overline{u}\overline{\rho^2\lambda}$                 | /      | 0      | ✓           |
| APEC               | Terashima et al. [14] | $\overline{\rho} \ \overline{u}$        | $\left(\overline{\rho e} - \frac{\delta^{+} \alpha  \delta^{+} \rho}{4}\right) \overline{u}$ | /      | 0      | ✓           |
| KEEP               | Kuya et al. [35]      | $\overline{\rho} \overline{u}$          | $\mathcal{F}_{\rho}\overline{e}$                                                             | ×      | ×      | ✓           |
| KEEP <sub>PE</sub> | Shima et al. [18]     | $\overline{\rho} \overline{u}$          | $\overline{\rho e} \ \overline{u}$                                                           | ✓      | ×      | ✓           |
| $KGP_{Pt}$         | Bernades et al. [23]  | $\overline{\rho} \overline{u}$          | -                                                                                            | ✓      | ✓      | ×           |
|                    |                       |                                         |                                                                                              |        |        |             |

Table 1: Summary of the compared numerical discretizations.  $\checkmark$ : property verified,  $\bigcirc$ : property verified approximately,  $\checkmark$ : property not verified, n.a. : not applicable.  $\lambda = (\partial e/\partial \rho)_p$ ,  $\alpha = (\partial \rho e/\partial \rho)_p$ ,  $\overline{\rho}^{\lambda} = \delta^+(\rho^2 \lambda)/\delta^+\alpha$ . Momentum and total energy fluxes are calculated according to Eq. (11).

### **6** Numerical results

In this section, we present numerical tests designed to assess the proposed discrete formulations. The focus in on verifying the correct fulfillment of the PEP property and the exact total-energy conservation in inviscid flows. To place the performance of the proposed schemes in context, we compare them against classical formulations from the literature; a summary of all considered schemes in real gases simulations is provided in Table 1. For the thermally perfect gas simulations, our formulation employs the mass flux in Eq. (32) and the internal energy flux contained in Eq. (31). To compare the performances of our newly derived schemes, we consider the KEEP scheme by Kuya et al. [35], commonly used for simulations of calorically perfect gases, and its variant KEEP<sub>PE</sub> by Shima et al. [18] which is PEP for calorically perfect gases; regarding formulations specifically designed for real-gas simulations we consider the Approximately Pressure Equilibrium Conserving (APEC) scheme developed by Terashima et al. [14] and the pressure-based KGP<sub>Pt</sub> scheme studied in Bernades et al. [23]. In all tests, time integration is carried out using the standard fourth-order Runge–Kutta method.

We consider the compressible Euler equations in dimensionless form: reference quantities are set as the standard ambient conditions for temperature and pressure (SATP), i.e.  $T_{\rm SATP}^* = 298.15 \, \rm K$  and  $p_{\rm SATP}^* = 1 \, \rm atm$ . This normalization enables a consistent treatment of different thermodynamic regimes and allows the equation of state to be adapted to conditions representative of each model. In particular, we examine thermally perfect gases at high enthalpy, van der Waals fluids at supercritical conditions, and Peng–Robinson fluids at both supercritical and transcritical regimes.

The assessment is carried out using two benchmark problems. A one-dimensional density wave is used to evaluate the PEP property and to compare the proposed schemes against classical counterparts. Additionally, a two-dimensional double-jet configuration is employed to investigate the schemes' behavior in a more demanding flow setting, studying both the energy conservation and the insurgence of non-physical pressure oscillations.

#### 6.1 One-dimensional density wave

The one-dimensional density wave is solved in the domain  $\Omega: x \in [0, L]$  with L=1, discretized in N=41 evenly spaced points (h=0.025). A fourth-order accurate spatial discretization is employed. To minimize contamination from time-integration errors, the CFL number has been set to  $5 \times 10^{-3}$  for all the considered cases. The initial conditions correspond to a smooth density perturbation convected at constant velocity, defined as

$$\begin{cases}
\rho(x,0) = \rho_0 \left[ A + Be^{\sin\left(\frac{2\pi x}{L}\right)} \right] \\\nu(x,0) = u_0 \\
p(x,0) = p_0
\end{cases}$$
(44)

with  $\{\rho_0,p_0\}=\{\rho_c,100\}$  for the van der Waals and Peng–Robinson models in the supercritical regime, while  $\{\rho_0,p_0\}=\{\rho_{\mathrm{SATP}},0.45\}$  for the thermally perfect one, where  $\rho_{\mathrm{SATP}}^*=1.795$  Kg/m³. Modulation constants are A=0.07 and B=0.12. The value  $u_0=1$  sets the reference time  $t_{\mathrm{ref}}=L/u_0=1$ , hence  $t=t^*$ .

![](_page_11_Figure_2.jpeg)

Figure 1: Global kinetic-energy evolution for the one-dimensional density wave test with various gas models.

To monitor the onset of numerical instabilities, we track the normalized variation of a generic quantity  $\phi$  defined as

$$\langle \phi \rangle = \frac{\int_{\Omega} \phi \, d\Omega - \int_{\Omega} \phi_0 \, d\Omega}{\int_{\Omega} \phi_0 \, d\Omega}.$$

In particular, we consider the kinetic energy  $\rho\kappa$ . Since both pressure and velocity are expected to remain constant for this problem, the kinetic energy should be preserved up to machine precision. Therefore, deviations in  $\langle \rho\kappa \rangle$  provide a sensitive indicator of scheme robustness.

Fig. 1 reports the evolution of  $\langle \rho \kappa \rangle$  for the schemes under consideration. All shown non-PEP schemes exhibit numerical blow-up. Among them, the first to become unstable is KEEP, which is not PEP even for ideal gases. This is followed by KEEP<sub>PE</sub>, which satisfies the PEP property only for calorically perfect gases, and by APEC, which is only approximately PEP for real-gas models. In contrast, EPEP-RG maintains the error on kinetic energy at machine-zero throughout the simulation. The APEP-RG scheme (not displayed) has performances almost indistinguishable from that of the EPEP-RG in this time interval. These trends are consistent across all the tested thermodynamic models: thermally perfect, van der Waals, and Peng–Robinson.

Further insight is provided in Fig. 2, which shows solution snapshots at a time when KEEP and KEEP<sub>PE</sub> have already become unstable, while APEC is still running. For EPEP-RG, the pressure remains equal to its initial value and the density closely matches the exact solution. On the other hand, APEC exhibits spurious oscillations in both pressure and density, indicating the onset of numerical degradation despite not having blown up yet.

For the Peng–Robinson model, the EPEP-RG, APEP-RG and KGP<sub>Pt</sub> simulations have been performed up to a final time t=100, without recording any instability. In Fig. 3, the density profile solution is reported at t=100, together with the time history of the total energy. Interestingly enough, the density profiles of the APEP-RG and KGP<sub>Pt</sub> formulations show similar dispersion features, a circumstance that should probably be attributed to the fact that the two formulations share the same mass and momentum fluxes. Fig. 3b shows that, as predicted, the global total energy remains constant up to machine precision in this long simulation for the EPEP-RG and APEP-RG schemes, whereas for the KGP<sub>Pt</sub> scheme a slow accumulation is present during the whole simulation, reaching a final value of the order  $10^{-3}$ .

To assess the performances of the newly proposed schemes in transcritical conditions, a simulation has been also carried out with the Peng–Robinson model for initial conditions  $\{A,B,\rho_0,p_0\}=\{0,2/3,\rho_c,135\}$ . In this transcritical case, in addition to the KEEP, APEC and KEEP<sub>PE</sub> formulations, also the EPEP-RG scheme shows strong instabilities, probably due to the near-singular behavior of the thermodynamic derivatives in the vicinity of the pseudo-critical line, where the main thermodynamic quantities—such as internal energy or the speed of sound—undergo rapid and large variations that can violate the assumptions on which the exact discrete formulation relies. Nevertheless, the APEP-RG formulation shows a good robustness, similar to that of the KGP<sub>Pt</sub> scheme, which is exactly PEP for this case. Note that for this transcritical case, which is particularly susceptible to instabilities, even the KGP<sub>Pt</sub> scheme eventually diverges at long times. In fact, a long simulation shows blow up for both the APEP-RG and KGP<sub>Pt</sub> schemes at  $t\approx 96$  and  $t\approx 107$ , respectively. Consequently, it seems that no numerical method currently available is capable of handling this test for sufficiently long times. This result could possibly be consistent with the previous observations about the instability near the pseudo-critical line as a consequence of the direct computation of the thermodynamic derivatives to solve the main system of equations. In fact, the KGP<sub>Pt</sub> involves the expression of the speed of sound, whose evaluation inherently requires several such derivatives, making the scheme susceptible to some ill-conditioning in the transcritical region. In Fig. 4, APEP-RG and KGP<sub>Pt</sub> density profiles and total energy evolutions are reported at t=50. APEP-RG

![](_page_12_Figure_2.jpeg)

Figure 2: Density and pressure profiles for the one-dimensional density wave test. (a)-(b): thermally perfect gas (t = 16), (c)-(d) van der Waals equation of state in supercritical conditions (t = 13), (e)-(f) Peng–Robinson equation of state in supercritical conditions (t = 14).

![](_page_13_Figure_2.jpeg)

Figure 3: Density profile (a) and total energy evolution (b) for the one-dimensional density wave test for the Peng-Robinson equation of state in supercritical conditions at t = 100.

![](_page_13_Figure_4.jpeg)

Figure 4: Density profile (a) and total energy evolution (b) for the one-dimensional density wave test for the Peng-Robinson equation of state in transcritical conditions at t = 50.

shows a slightly better agreement with the exact solution, while KGP<sub>Pt</sub> exhibits a lack of total energy conservation, with an error accumulating during the evolution and reaching a final value of approximately  $6 \times 10^{-4}$ .

# 6.2 Two-dimensional inviscid double-jet flow at high-enthalpy and supercritical conditions

The two-dimensional, inviscid double-jet flow is simulated in the rectangular domain  $(x,y) \in [0,L] \times [-L/4,L/4]$ , with L=1, discretized in  $N_x \times N_y = 65 \times 33$  evenly spaced points with second-order accurate spatial discretizations, with CFL set as 0.01 to constrain time-integration errors. The flow is initialized as

$$\begin{cases}
 \{ u(x,y,0) = A_u \left[ 1 + A_u \tanh \left( \theta(y + L/10) \right) \right] & \text{for } y \leq 0 \\
 T(x,y,0) = aA_t \left[ 3/2 - A_t \tanh \left( \theta(y + L/10) \right) \right] & \text{for } y \leq 0 
\end{cases} \\
 \{ u(x,y,0) = A_u \left[ 1 - A_u \tanh \left( \theta(y - L/10) \right) \right] & \text{for } y > 0 \\
 T(x,y,0) = aA_t \left[ 3/2 + A_t \tanh \left( \theta(y - L/10) \right) \right] & \text{for } y > 0 
\end{cases} \\
 v(x,y,0) = \varepsilon \sin \left( 2m\pi x/L \right) \\
 p(x,y,0) = p_0
\end{cases}$$

![](_page_14_Figure_2.jpeg)

Figure 5: Two-dimensional inviscid double-jet flow for a thermally perfect gas. (a) pressure profiles evaluated at y=0.33 and  $t/t_{\rm ref}=2.5$ , (b) pressure profiles evaluated at x=0 and  $t/t_{\rm ref}=2.5$ , (c) total energy evolution, (d) entropy evolution.

with the velocity field being the same for each gas model, given  $\{A_u, \varepsilon, m\} = \{1/2, 0.05, 3\}$ , to have a transversal jet that generates three roll-up vortices. Temperature and pressure fields are, on the other hand, established with respect to the specific equation of state. Thus, we set  $\{a, A_t, p_0\} = \{2.6, 2/3, 0.1\}$  for the thermally perfect case and  $\{a, A_t, p_0\} = \{2.5, 1/2, 150\}$  for the supercritical test, carried out by means of the van der Waals model. Finally, the parameter  $\theta = 30$  represents the thickness of the shear layer, and a reference time is defined as  $t_{\rm ref} = m^{-1}/\max_{x,y} u(x,y,0) \approx 0.445$ .

As the numerical mass flux defined in Eq. (25) is singular in the wide, initially uniform regions, we decided to use only the APEP-RG formulation for this test case, compared against the KEEP<sub>PE</sub>, APEC, and KGP<sub>Pt</sub> schemes, which have shown better performances in the previous 1D test case with respect to the KEEP scheme. In Fig. 5, the results of the high-enthalpy case are reported, in terms of pressure profiles along horizontal and vertical lines at y = 0.33 and x = 0, respectively (Fig. 5a and 5b), together with the time evolution of the total energy and entropy. Although the results are generally satisfactory for all the schemes considered, Fig. 5b shows that KEEP<sub>PE</sub> and APEC are starting to exhibit point-to-point oscillations in the vertical profile of pressure. The total energy evolution depicted in Fig. 5cconfirms that the KGP<sub>Pt</sub> scheme steadily deviates from exact conservation, whereas the global entropy evolution reported in Fig. 5d shows that all the considered formulations violate the exact entropy preservation, as none of them is exactly entropy conservative. The pressure oscillations highlighted for the KEEP<sub>PE</sub> and APEC schemes in Fig. 5b are much more visible in the supercritical case computed with the van der Waals model, reported in Fig. 6a and 6b. In this simulation the APEP and KGP<sub>Pt</sub> schemes remain essentially free of oscillation.

![](_page_15_Figure_2.jpeg)

Figure 6: Two-dimensional inviscid double-jet flow for a van der Waals gas at supercritical conditions. (a) pressure profiles evaluated at y=0.33 and  $t/t_{\rm ref}=2.5$ , (b) pressure profiles evaluated at x=0 and  $t/t_{\rm ref}=2.5$ , (c) total energy evolution, (d) entropy evolution.

# 6.3 Two-dimensional inviscid double-jet flow at transcritical conditions

Transcritical conditions necessitate a more detailed analysis due to the different behavior of thermodynamic derivatives near the critical point, especially when discretizing the energy equation which requires the computation of internal energy and its gradients in the thermodynamic space. In this section, the inviscid double-jet flow is simulated onto the same grid and within the same numerical setup presented in Section 6.2. Initial conditions also share the same symbolic form, with the  $\{a,A_t,p_0\}=\{2,1/2,180\}$ . This ensures a dimensional temperature  $T^*\in[\sim 298.8,\sim 587.9]\,\mathrm{K}$ , therefore crossing the critical temperature for  $\mathrm{CO}_2,\,T_c^*=304.12\,\mathrm{K}$ . Corresponding dimensional pressure is  $p^*\approx 2.43\times p_c^*=2.43\times 73.8\,\mathrm{atm}$ . This time, the Peng–Robinson model has been used to carry out the simulations. Figures 7a and 7b report the usual pressure profiles at t=1.6. In this case, the pressure oscillations are much more evident for the KEEP<sub>PE</sub> and APEC schemes, even if the simulation is stopped at an earlier time. The total energy evolution is consistent with the other simulations, whereas the entropy evolution reported in Fig. 7d shows an oscillating behavior, which is exacerbated for the KGP<sub>Pt</sub> formulation. Fig. 8 reports a snapshot of the two-dimensional pressure field as calculated by the various formulations, confirming the contamination of the solution due to the growing oscillations in the KEEP<sub>PE</sub> and APEC schemes.

![](_page_16_Figure_2.jpeg)

Figure 7: Two-dimensional inviscid double-jet flow for a Peng–Robinson gas at transcritical conditions. (a) pressure profiles evaluated at y = 0.33 and t/tref = 2.5, (b) pressure profiles evaluated at x = 0.5 and t/tref = 2.5, (c) total energy evolution, (d) entropy evolution.

# 7 Conclusions

As highlighted throughout this work, the numerical simulation of compressible flows for non-ideal fluids is frequently challenged by the generation of spurious pressure oscillations. While addressing this issue requires numerical methods that satisfy the pressure-equilibrium-preserving (PEP) property, ensuring strict PEP compliance for general equations of state has historically proven difficult. Previous attempts often compromise the exact discrete conservation of total energy, a property that remains essential for maintaining physical consistency and correctly capturing shocks.

In this work, we demonstrate that the exact discrete conservation of mass, momentum, and total energy is not mutually exclusive with the PEP condition. Although the present mathematical framework is derived in the context of finitedifference discretizations, the resulting two-point numerical fluxes are highly versatile. They can be seamlessly applied to other spatial discretization frameworks, such as structured and unstructured finite volume or discontinuous Galerkin formulations.

Our main contribution is the development of a fully conservative and exactly pressure-equilibrium-preserving scheme, denoted as Exact PEP (EPEP-RG). We derive a generic formula applicable to any arbitrary equation of state and provide specific, computationally viable flux formulations for thermally perfect, van der Waals, and Peng–Robinson gases. Building upon this theoretical foundation, and recognizing the complexities inherent to certain thermodynamic regimes, we also propose a robust practical alternative: the Approximate PEP (APEP-RG) scheme. The APEP-RG

![](_page_17_Figure_2.jpeg)

Figure 8: Instantaneous pressure fields for the two-dimensional inviscid double-jet flow for a Peng–Robinson gas at transcritical conditions and t/tref = 1.6.

formulation strictly maintains full primary conservation, including kinetic-energy preservation by convection, while enforcing the PEP condition in an approximate sense.

We have validated the proposed schemes using rigorous numerical benchmarks. In the density wave advection test, the EPEP-RG scheme is able to successfully preserve pressure equilibrium across a variety of EoS. This behavior contrasts favorably with standard formulations from the literature, which either become unstable or fail to conserve total energy. Furthermore, simulations of a compressible mixing layer confirm the exact conservation of total energy alongside the elimination of spurious oscillations in the pressure field, effectively addressing the energy conservation limitations of previous non-oscillatory schemes. While the exact EPEP-RG formulation can encounter numerical singularities due to problematic thermodynamic derivatives in transcritical regimes, the APEP-RG scheme reliably circumvents these issues. Overall, the proposed schemes demonstrate highly favorable robustness, stability, and accuracy. The APEP-RG framework, in particular, emerges as a simple and resilient tool capable of managing severe thermodynamic nonlinearities—including transcritical phenomena—without sacrificing primary conservation or numerical stability.

Despite these successes, certain limitations remain that pave the way for future research. The EPEP-RG scheme currently requires a specialized fix to properly handle the singularities of thermodynamic derivatives across phase boundaries in transcritical regimes or in areas where thermodynamic quantities are nearly uniform. The APEP-RG scheme provides a robust workaround for these issues, while also introducing a degree of flexibility regarding the choice of the density averaging operator. Future work could systematically explore the characteristics of various averaging operators to identify the optimal formulation, potentially exploiting this degree of freedom to impart additional structure-preserving properties to the scheme, such as discrete entropy conservation or stability.

Further efforts will focus on extending this framework to multi-component and multi-phase flow formulations, as well as evaluating the schemes' performances on more complex, physically realistic configurations, such as wall-bounded turbulent flows. Ultimately, the conservative and equilibrium-preserving methodologies introduced herein represent a significant step forward toward a unified, robust framework for the high-fidelity simulation of complex real-gas flows.

# References

J. Comput. Phys. 300 (2015) 116–135.

J. Comput. Phys. 478 (2023) 111973.

J. Comput. Phys. 427 (2021) 110060.

- [1] R. Pecnik, E. Rinaldi, P. Colonna, Computational fluid dynamics o ˇ [f a radial compressor operating with supercritical CO](https://doi.org/10.1115/1.4007196)2, J. Eng. Gas Turbines Power 134 (2012).
- [2] L. Jofre, J. Urzay, [Transcritical diffuse-interface hydrodynamics of propellants in high-pressure combustors of](https://doi.org/10.1016/j.pecs.2020.100877) chemical propulsion Prog. Energy Combust. Sci. 82 (2021) 100877.
- [3] A. Guardone, P. Colonna, M. Pini, A. Spinelli, [Nonideal compressible fluid dynamics of dense vapors and supercritical fluids,](https://doi.org/10.1146/annurev-fluid-120720-033342) Annu. Rev. Fluid Mech. 56 (2024) 241–269.
- [4] H. Terashima, M. Koshi, [Approach for simulating gas–liquid-like flows under supercritical pressures using a high-order central differencing](https://doi.org/10.1016/j.jcp.2012.06.021) J. Comput. Phys. 231 (2012) 6907–6923.
- [5] R. Abgrall, [How to prevent pressure oscillations in multicomponent flow calculations: A quasi conservative approach,](https://doi.org/10.1006/jcph.1996.0085) J. Comput. Phys. 125 (1996) 150–160.
- [6] T. Y. Hou, P. G. LeFloch, [Why nonconservative schemes converge to wrong solutions: error analysis,](https://doi.org/10.1090/s0025-5718-1994-1201068-0) Math. Comput. 62 (1994) 497–530.
- [7] S. Karni, [Multicomponent flow calculations by a consistent primitive algorithm,](https://doi.org/10.1006/jcph.1994.1080) J. Comput. Phys. 112 (1994) 31–43.
- [8] K.-M. Shyue, [A fluid-mixture type algorithm for compressible multicomponent flow with Mie–Grüneisen equation of state,](https://doi.org/10.1006/jcph.2001.6801)
- J. Comput. Phys. 171 (2001) 678–707. [9] S. Kawai, H. Terashima, H. Negishi, A robust and accurate [numerical method for transcritical turbulent flows at supercritical pressure](https://doi.org/10.1016/j.jcp.2015.07.047)
- [10] G. Lacaze, T. Schmitt, A. Ruiz, J. C. Oefelein, [Comparison of energy-, pressureand enthalpy-based approaches for](https://doi.org/10.1016/j.compfluid.2019.01.002) modeling supercritical Comput. Fluids 181 (2019) 35–56.

[14] H. Terashima, N. Ly, M. Ihme, [Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible](https://doi.org/https://doi.org/10.1016/j.jcp.2024.113701)

- [11] R. Abgrall, S. Karni, [Computations of compressible multifluids,](https://doi.org/10.1006/jcph.2000.6685) J. Comput. Phys. 169 (2001) 594–623.
- [12] P. C. Ma, Y. Lv, M. Ihme, An entropy-stable hybrid scheme [for simulations of transcritical real-fluid flows,](https://doi.org/10.1016/j.jcp.2017.03.022) J. Comput. Phys. 340 (2017) 330–357.
- [13] Y. Fujiwara, Y. Tamaki, S. Kawai, [Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component](https://doi.org/10.1016/j.jcp.2023.111973)
- J. Comput Phys. 524 (2025) 113701.
- [15] W. J. Feiereisen, W. C. Reynolds, J. H. Ferziger, Numerical Simulation of Compressible, Homogeneous Turbulent Shear Flow, Technical Report TF-13, Stanford University, 1981.
- [16] A. Jameson, The construction of discretely conservative finite volume schemes that also globally conserve energy or entropy, J. Sci. Comput. 34 (2008) 152–187.
- [17] S. Pirozzoli, [Generalized conservative approximations of split convective derivative operators,](https://doi.org/10.1016/j.jcp.2010.06.006) J. Comput. Phys.
- 229 (2010) 7180–7190. [18] N. Shima, Y. Kuya, Y. Tamaki, S. Kawai, [Preventing spurious pressure oscillations in split convective form discretization for compressible](https://doi.org/10.1016/j.jcp.2020.110060)
- [19] H. Ranocha, G. Gassner, [Preventing pressure oscillations does not fix local linear stability issues of entropy-based split-form high-order](https://doi.org/10.1007/s42967-021-00148-z) Commun. Appl. Math. Comput. 4 (2022) 880–903.
- [20] C. De Michele, G. Coppola, [Asymptotically entropy-conservative and kinetic-energy preserving numerical fluxes for compressible](https://doi.org/10.1016/j.jcp.2023.112439) J. Comput. Phys. 492 (2023) 112439.
- [21] C. De Michele, G. Coppola, Novel pressure-equilibrium [and kinetic-energy preserving fluxes for compressible flows](https://doi.org/https://doi.org/10.1016/j.jcp.2024.113338) based on the harmonic J. Comput. Phys. 518 (2024) 113338.
- [22] S. Kawai, S. Kawai, [Logarithmic mean approximation in improving entropy conservation in KEEP scheme with pressure](https://doi.org/10.1016/j.jcp.2025.113897) equilibrium J. Comput. Phys. 530 (2025) 113897.
- [23] M. Bernades, L. Jofre, F. Capuano, Kinetic-energyand [pressure-equilibrium-preserving schemes for real-gas turbulence in the transcritical](https://doi.org/10.1016/j.jcp.2023.112477) J. Comput. Phys. 493 (2023) 112477.
- [24] S. Pirozzoli, [Stabilized non-dissipative approximations of Euler equations in generalized curvilinear coordinates,](https://doi.org/https://doi.org/10.1016/j.jcp.2011.01.001) J. Comput. Phys. 230 (2011) 2997–3014.
- [25] Y. Kuya, S. Kawai, High-order accurate kinetic-energy [and entropy preserving \(KEEP\) schemes on curvilinear grids,](https://doi.org/10.1016/j.jcp.2021.110482) J. Comput. Phys. 442 (2021) 110482.

- [26] P. LeFloch, J. Mercier, C. Rohde, [Fully discrete, entropy conservative schemes of arbitrary order,](https://doi.org/10.1137/s003614290240069x) SIAM J. Numer. Anal. 40 (2002) 1968–1992.
- [27] T. C. Fisher, M. H. Carpenter, [High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains,](https://doi.org/10.1016/j.jcp.2013.06.014) J. Comput. Phys. 252 (2013) 518–557.
- [28] H. Ranocha, [Comparison of some entropy conservative numerical fluxes for the Euler equations,](https://doi.org/10.1007/s10915-017-0618-1) J. Sci. Comput. 76 (2018) 216–242.
- [29] A. Aiello, C. De Michele, G. Coppola, [Entropy conservative discretization of compressible Euler equations with an](https://doi.org/10.1016/j.jcp.2025.113836) arbitrary equation J. Comput. Phys. 528 (2025) 113836.
- [30] G. Coppola, F. Capuano, L. de Luca, [Discrete energy-conservation properties in the numerical solution of the Navier–Stokes equations](https://doi.org/10.1115/1.4042820) Appl. Mech. Rev. 71 (2019) 010803–1 – 010803–19.
- [31] G. Coppola, F. Capuano, S. Pirozzoli, L. de Luca, [Numerically stable formulations of convective terms for turbulent compressible flo](https://doi.org/10.1016/j.jcp.2019.01.007) J. Comput. Phys. 382 (2019) 86–104.
- [32] A. E. P. Veldman, [A general condition for kinetic-energy preserving discretization of flow transport equations,](https://doi.org/10.1016/j.jcp.2019.108894) J. Comput. Phys. 398 (2019) 108894.
- [33] G. Coppola, A. E. P. Veldman, [Global and local conservation of mass, momentum and kinetic energy in the simulation of](https://doi.org/10.1016/j.jcp.2022.111879) compressible J. Comput. Phys. 475 (2023) 111879.
- [34] C. De Michele, G. Coppola, [Numerical treatment of the energy equation in compressible flows simulations,](https://doi.org/10.1016/j.compfluid.2022.105709) Comput. Fluids 250 (2023) 105709.
- [35] Y. Kuya, K. Totani, S. Kawai, Kinetic energy and entropy [preserving schemes for compressible flows by split convective forms,](https://doi.org/10.1016/j.jcp.2018.08.058) J. Comput. Phys. 375 (2018) 823–853.

[36] A. Aiello, C. De Michele, G. Coppola, [Formulation of entropy-conservative discretizations for compressible flows](https://doi.org/10.48550/arXiv.2507.08115) of thermally perfect

[40] C. De Michele, A. K. Edoh, G. Coppola, [Finite-difference compatible entropy-conserving schemes for the compressible Euler equations](https://doi.org/10.1016/j.jcp.2025.114262)

[42] R. Klein, B. Sanderse, P. Costa, R. Pecnik, R. Henkes, [Generalized Tadmor conditions and structure-preserving numerical fluxes for](https://doi.org/10.48550/arXiv.2603.15112)

- arXiv:2507.08115 [physics.flu-dyn] (2025).
- [37] E. Tadmor, [The numerical viscosity of entropy stable schemes for systems of conservation laws. I,](https://doi.org/10.1090/s0025-5718-1987-0890255-3) Math. Comput. 179 (1987) 91–103.
- [38] E. Tadmor, [Entropy stability theory for difference approximations of nonlinear conservation laws and related time-dependent problems](https://doi.org/10.1017/s0962492902000156)

Acta Numerica 12 (2003) 451–512.

- [39] H. Ranocha, [Entropy conserving and kinetic energy preserving numerical methods for the Euler equations using summation-by-parts](https://doi.org/10.1007/978-3-030-39647-3_42) in: S. J. Sherwin, D. Moxey, J. Peiró, P. E. Vincent, C. Schwab (Eds.), Spectral and High Order Methods for Partial Differential Equations ICOSAHOM 2018. Lecture Notes in Computational Science and Engineering, volume 134, Springer, Cham, 2020, p. 525 – 535. doi:[10.1007/978-3-030-39647-3\\_42](http://dx.doi.org/10.1007/978-3-030-39647-3_42).
- J. Comput. Phys. (2025) 114262.
- [41] F. Ismail, P. L. Roe, [Affordable, entropy-consistent Euler flux functions II: Entropy production at shocks,](https://doi.org/10.1016/j.jcp.2009.04.021) J. Comput. Phys. 228 (2009) 5410–5436.
- 2026. doi:[10.48550/arXiv.2603.15112](http://dx.doi.org/10.48550/arXiv.2603.15112). [arXiv:2603.15112](http://arxiv.org/abs/2603.15112). [43] M. A. Hansen, T. C. Fisher, Entropy Stable Discretization of Compressible Flows in Thermochemical Nonequi-
- librium, Technical Report, Sandia National Lab. (SNL-NM), Albuquerque, NM (United States), 2019. URL: <https://www.osti.gov/biblio/1763209>. doi:[10.2172/1763209](http://dx.doi.org/10.2172/1763209).
- [44] M. Chase, NIST-JANAF Thermochemical Tables, 4th Edition, American Institute of Physics, -1, 1998. URL: <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf>.
- [45] B. J. McBride, M. J. Zehe, S. Gordon, NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species, Technical Report, Glenn Research Center, Cleveland, Ohio, 2002. URL: <https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf>.
- [46] I. Tosun, The Thermodynamics of Phase and Reaction Equilibria, second edition, Elsevier B.V., 2021.